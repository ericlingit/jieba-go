package tokenizer

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
)

const minFloat float64 = -3.14e100

var zh = regexp.MustCompile(`\p{Han}+`)
var alnum = regexp.MustCompile(`([a-zA-Z0-9]+)`)

var stateChange = map[string][]string{
	"B": {"E", "S"}, // E->B, S->B
	"M": {"B", "M"},
	"E": {"B", "M"},
	"S": {"E", "S"},
}

type textBlock struct {
	id        int
	text      string
	doProcess bool
}

type resultBlock struct {
	id     int
	tokens []string
}

type tailProba struct {
	index int
	proba float64
}

type transitionRoute struct {
	from  string
	proba float64
}

type Tokenizer struct {
	ready bool
	pd    prefixDictionary
	hmm   hiddenMarkovModel
	// Values below are for debugging.
	dag      map[int][]int
	dagProba map[int][]tailProba
}

func NewTokenizer(dictionaryFile string) *Tokenizer {
	tk := Tokenizer{}
	tk.pd = *newPrefixDictionaryFromFile(dictionaryFile)
	tk.hmm = newJiebaHMM()
	tk.ready = true
	return &tk
}

func NewJiebaTokenizer() *Tokenizer {
	tk := Tokenizer{}
	tk.pd = *newJiebaPrefixDictionary()
	tk.hmm = newJiebaHMM()
	tk.ready = true
	return &tk
}

// Perform Cut in worker goroutines in parallel.
// If ordered is true, the returned slice will be sorted
// according to the order of the input text. Sorting will
// adversely impact performance by approximately 30%.
func (tk *Tokenizer) CutParallel(text string, hmm bool, numWorkers int, ordered bool) []string {
	tk.pd.lock.RLock()
	defer tk.pd.lock.RUnlock()
	// Split text into zh and non-zh blocks.
	blocks := make(chan textBlock, len(text))
	zhIndexes := zh.FindAllIndex([]byte(text), -1)
	go func() {
		defer close(blocks)
		for _, block := range splitText(text, zhIndexes) {
			blocks <- block
		}
	}()
	// Launch worker goroutines that fetch work from `blocks`.
	// Completed work are sent to `result`.
	result := make(chan resultBlock, len(text))
	stop := make(chan struct{})
	defer close(stop)
	wg := sync.WaitGroup{}
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go func() {
			tk.worker(blocks, stop, result, hmm)
			wg.Done()
		}()
	}
	go func() {
		defer close(result)
		wg.Wait()
	}()
	if ordered {
		// Collect `resultBlock` from `result`.
		rblocks := []resultBlock{}
		for rb := range result {
			rblocks = append(rblocks, rb)
		}
		// Sort rblocks.
		sort.Slice(rblocks, func(i, j int) bool {
			return rblocks[i].id < rblocks[j].id
		})
		// Extract strings.
		tokens := []string{}
		for _, rb := range rblocks {
			tokens = append(tokens, rb.tokens...)
		}
		return tokens
	} else {
		// Collect `resultBlock` from `result` and extract
		// string tokens.
		tokens := []string{}
		for rb := range result {
			tokens = append(tokens, rb.tokens...)
		}
		return tokens
	}
}

// Worker for CutParallel() method.
// A worker fetches work from `blocks` channel, processes the
// block, and sends the result to the `result` channel.
func (tk *Tokenizer) worker(blocks chan textBlock, stop chan struct{}, result chan resultBlock, hmm bool) {
	for b := range blocks {
		select {
		case <-stop:
			return
		case result <- resultBlock{b.id, tk.cutBlock(b, hmm)}:
		}
	}
}

// Cut text and return a slice of tokens.
func (tk *Tokenizer) Cut(text string, useHmm bool) []string {
	tk.pd.lock.RLock()
	defer tk.pd.lock.RUnlock()
	zhIndexes := zh.FindAllIndex([]byte(text), -1)
	blocks := splitText(text, zhIndexes)

	result := []string{}
	for _, block := range blocks {
		result = append(result, tk.cutBlock(block, useHmm)...)
	}
	return result
}

// Identify the text index ranges to process.
func splitText(text string, markedIndexes [][]int) []textBlock {
	if len(markedIndexes) == 0 {
		return []textBlock{{0, text, false}}
	}

	// Find all in-between indexes.
	// For example, if text length is 15, and markedIndexes is as
	// follows:
	//     [][]int{
	//         {4, 6},
	//         {8, 10},
	//     }
	//
	// then we expect this result:
	//     [][]int{
	//         {0, 4},
	//         {4, 6}, // Marked index pairs (doProcess=true)
	//         {6, 8},
	//         {8, 10}, // Marked index pairs (doProcess=true)
	//         {10, 15},
	//     }
	count := 0
	blocks := []textBlock{}
	prevTail := 0
	for i, pair := range markedIndexes {
		// Pair has left-side gap.
		if pair[0] != prevTail {
			// Fill in the gap.
			filler := text[prevTail:pair[0]]
			blocks = append(blocks, textBlock{count, filler, false})
			count++
		}
		markedText := text[pair[0]:pair[1]]
		blocks = append(blocks, textBlock{count, markedText, true})
		prevTail = pair[1]
		count++

		// Last pair with a right-side gap.
		if i == len(markedIndexes)-1 && pair[1] != len(text) {
			// Fill in the gap.
			filler := text[pair[1]:]
			blocks = append(blocks, textBlock{count, filler, false})
		}
	}
	return blocks
}

func (tk *Tokenizer) cutBlock(block textBlock, hmm bool) []string {
	if block.doProcess {
		return tk.cutZh(block.text, hmm)
	}
	return tk.cutNonZh(block.text)
}

// cutZh `text` using a prefix dictionary, and optionally use a
// Hidden Markov model to identify and segment unknown words.
func (tk *Tokenizer) cutZh(text string, hmm bool) []string {
	dag := tk.buildDAG(text)
	dagPath := tk.findDAGPath(text, dag)
	dagPieces := tk.cutDAG(text, dagPath)
	if !hmm {
		return dagPieces
	}

	// Use HMM to segment uncut chars in dagPieces.
	words := []string{}
	uncutRunes := []rune{}
	for i, piece := range dagPieces {
		pieceRune := []rune(piece)
		// Collect singletons for HMM segmentation
		if len(pieceRune) == 1 {
			uncutRunes = append(uncutRunes, pieceRune[0])
			// Run cutHMM at the end of iteration only if there
			// are uncut runes.
			if i+1 >= len(dagPieces) && len(uncutRunes) != 0 {
				v := tk.hmm.viterbi(string(uncutRunes))
				newWords := tk.cutHMM(string(uncutRunes), v)
				words = append(words, newWords...)
				uncutRunes = nil
			}
		} else {
			// Run cutHMM when a length > 1 rune is encountered.
			if len(uncutRunes) != 0 {
				v := tk.hmm.viterbi(string(uncutRunes))
				newWords := tk.cutHMM(string(uncutRunes), v)
				words = append(words, newWords...)
				uncutRunes = nil
			}
			words = append(words, piece)
		}
	}
	return words
}

// Build a DAG out of every rune:rune+N piece from text string.
// The returned DAG's index values are based on []rune(text).
func (tk *Tokenizer) buildDAG(text string) map[int][]int {
	// Get the index of RUNES that are found in the prefix
	// dictionary. If not found, save the rune slice as is.
	textRunes := []rune(text)
	pieces := [][2]int{}
	for i, iRune := range textRunes {
		count, found := tk.pd.termFreq[string(iRune)]
		if !found || count == 0 {
			pieces = append(pieces, [2]int{i, i + 1})
			continue
		}
		for j := range textRunes[i:] {
			part := textRunes[i : j+1+i]
			val, found := tk.pd.termFreq[string(part)]
			if !found {
				break
			}
			if val > 0 {
				pieces = append(pieces, [2]int{i, j + 1 + i})
			}
		}
	}
	// fmt.Println("pieces:", pieces)

	dag := make(map[int][]int, len(textRunes))
	for _, p := range pieces {
		val, found := dag[p[0]]
		if !found {
			dag[p[0]] = []int{p[1]}
		} else {
			dag[p[0]] = append(val, p[1])
		}
	}
	// fmt.Println("dag:", dag)
	tk.dag = dag
	return dag
}

// Calculate the log probability of each DAG path (piece),
// and return the best path for each rune in `text`.
// The return value's index are based on []rune(text).
func (tk *Tokenizer) findDAGPath(text string, dag map[int][]int) [][2]int {
	total := math.Log(float64(tk.pd.size))
	textRunes := []rune(text)
	dagProba := make(map[int][]tailProba, len(textRunes))

	// Iterate through `textRunes` in reverse.
	for i := len(textRunes) - 1; i >= 0; i-- {
		// fmt.Printf("%q\n", string(textRunes[i]))
		dagProba[i] = []tailProba{}
		for _, j := range dag[i] {
			// Calculate current piece's probability.
			// piece_frequency = log(prefix_dictionary.get(piece) or 1.0) - total
			// piece_proba = piece_frequency + next_piece_proba
			tf := 1.0
			if val, found := tk.pd.termFreq[string(textRunes[i:j])]; found {
				tf = float64(val)
			}
			pieceFreq := math.Log(tf) - total

			// Get next piece's probability.
			nextPiece := []tailProba{{j, 0.0}}
			if val, found := dagProba[j]; found {
				nextPiece = val
			}
			// There could be more than 1 nextPiece, use the one
			// with the highest log probability.
			nextBestPiece := tk.maxIndexProba(nextPiece)
			pieceProba := pieceFreq + nextBestPiece.proba
			dagProba[i] = append(dagProba[i], tailProba{j, pieceProba})
			// fmt.Printf(
			// 	"  %q dagProba[%d][%d] = %f (%f %sFreq  + %f %sProba dagProba[%d][%d])\n",
			// 	string(textRunes[i:j]),
			// 	i,
			// 	j,
			// 	pieceProba,
			// 	pieceFreq,
			// 	string(textRunes[i:j]),
			// 	nextBestPiece.Proba,
			// 	string(textRunes[j:x]),
			// 	j,
			// 	x,
			// )
		}
	}
	tk.dagProba = dagProba
	// Keep paths with the highest log probability.
	return tk.findBestPath(text, dagProba)
}

// Return the map key whose float value is the highest.
func (tk *Tokenizer) maxIndexProba(items []tailProba) tailProba {
	prev := tailProba{-1, minFloat}
	best := tailProba{-1, minFloat}
	for _, item := range items {
		if item.proba >= prev.proba {
			best = item
		}
		prev = item
	}
	if best.index == -1 {
		return prev
	}
	return best
}

// Find the path with the highest probability.
// This is a helper method for findDAGPath().
func (tk *Tokenizer) findBestPath(text string, dagProba map[int][]tailProba) [][2]int {
	textRunes := []rune(text)

	bestPath := [][2]int{}
	for i := 0; i < len(textRunes) && i >= 0; {
		tail := tk.maxIndexProba(dagProba[i])
		bestPath = append(bestPath, [2]int{i, tail.index})
		i = tail.index
	}
	return bestPath
}

// Cut `text` using a DAG path built from a prefix dictionary.
func (tk *Tokenizer) cutDAG(text string, dagPath [][2]int) []string {
	textRune := []rune(text)
	pieces := []string{}
	for _, dagIndex := range dagPath {
		p := string(textRune[dagIndex[0]:dagIndex[1]])
		pieces = append(pieces, p)
	}
	return pieces
}

// Cut `text` according the the path found by the Viterbi algorithm.
func (tk *Tokenizer) cutHMM(text string, viterbiPath []string) []string {
	textRune := []rune(text)
	pieces := []string{}
	pieceStart := 0
	for i, state := range viterbiPath {
		pieceEnd := i + 1
		if state == "E" || state == "S" {
			pieces = append(pieces, string(textRune[pieceStart:pieceEnd]))
			pieceStart = pieceEnd
		}
	}
	return pieces
}

// Perform simple segmentation for space delimited alphanumeric
// words. All other characters are broken into individual runes.
func (tk *Tokenizer) cutNonZh(text string) []string {
	alnumIdx := alnum.FindAllIndex([]byte(text), -1)
	if len(alnumIdx) == 0 {
		return []string{}
	}

	textPieces := []string{}
	blocks := splitText(text, alnumIdx)
	for _, b := range blocks {
		if b.doProcess {
			textPieces = append(textPieces, b.text)
		} else {
			for _, r := range b.text {
				if unicode.IsSpace(r) {
					continue
				}
				textPieces = append(textPieces, string(r))
			}
		}
	}
	return textPieces
}

/*Build a prefix dictionary from `dictionaryLines`.

The dictionaryLines is a slice of strings that has the
vocabularies for segmentation. Each line contains the
information for one phrase/word, its frequency, and its
part-of-speech tag. Each field is separated by space.

For example:

	AT&T 3 v
	今天 2 n
	大學 4 n

This function returns a prefix dictionary that contains each
phrase/word's prefix.

For example:
{
	"A":    0,
	"AT":   0,
	"AT&":  0,
	"AT&T": 3,
	"今":   0,
	"今天":  2,
	"大":   0,
	"大學":  4,
}
*/
func (tk *Tokenizer) buildPrefixDictionary(dictionaryLines []string) error {
	tk.pd.termFreq = make(map[string]int, len(dictionaryLines)*2)
	total := 0
	for _, line := range dictionaryLines {
		parts := strings.SplitN(line, " ", 3)
		word := parts[0]
		count, err := strconv.Atoi(parts[1])
		if err != nil {
			return err
		}
		total += count
		tk.pd.termFreq[word] = count

		// Add word pieces.
		wordR := []rune(word)
		piece := ""
		for _, char := range wordR[:len(wordR)-1] {
			piece += string(char)
			_, found := tk.pd.termFreq[piece]
			if !found {
				tk.pd.termFreq[piece] = 0
			}
		}
	}
	tk.pd.size = total
	return nil
}

// Add a word to the prefix dictionary.
// If word already exists, the word's frequency value will
// be updated. If freq is less than 1, a frequency will be
// automatically calculated.
func (tk *Tokenizer) AddWord(word string, freq int) {
	if freq < 1 {
		freq = tk.pd.suggestFreq(word, tk)
	}
	tk.pd.lock.Lock()
	defer tk.pd.lock.Unlock()
	tk.pd.addTerm(word, freq)
}

type prefixDictionary struct {
	termFreq map[string]int
	size     int
	ready    bool
	lock     sync.RWMutex
	source   string
}

func newPrefixDictionaryFromFile(filename string) *prefixDictionary {
	pd := prefixDictionary{}
	pd.lock.Lock()
	defer pd.lock.Unlock()

	pd.source = filename
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// From file size, calculate how much map space to pre-allocate.
	// Each line takes, on average, 14.5 bytes.
	fileInfo, err := file.Stat()
	if err != nil {
		log.Fatal(err)
	}
	pd.termFreq = make(map[string]int, fileInfo.Size()/14)
	// Scan and parse line by line.
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, " ", 3)
		word := parts[0]
		count, err := strconv.Atoi(parts[1])
		if err != nil {
			log.Fatal(err)
		}
		// Source file may contain duplicates.
		_, found := pd.termFreq[word]
		if !found {
			pd.termFreq[word] = count
			pd.size += count
		}
		// // Add word fragments.
		// wordR := []rune(word)
		// piece := ""
		// for _, char := range wordR[:len(wordR)-1] {
		// 	piece += string(char)
		// 	_, found := pd.termFreq[piece]
		// 	if !found {
		// 		pd.termFreq[piece] = 0
		// 	}
		// }
	}
	pd.ready = true
	return &pd
}

func newJiebaPrefixDictionary() *prefixDictionary {
	// Load pre-built prefix dictionary from gob file.
	gobFile, err := os.Open("prefix_dictionary.gob")
	if err != nil {
		log.Fatalf("failed to open gob file: %v", err)
	}
	defer gobFile.Close()

	pd := prefixDictionary{}
	pd.lock.Lock()
	defer pd.lock.Unlock()
	decoder := gob.NewDecoder(gobFile)
	if err := decoder.Decode(&pd.termFreq); err != nil {
		log.Fatalf("failed to decode pfDict from gobFile: %v", err)
	}
	pd.size = 60_101_967
	pd.ready = true
	pd.source = "prefix_dictionary.gob"
	return &pd
}

func (pd *prefixDictionary) addTerm(term string, freq int) {
	pd.lock.Lock()
	defer pd.lock.Unlock()
	pd.termFreq[term] = freq
	pd.size += freq
}

// Calculate a frequency value based on current prefix
// dictionary and its total size.
func (pd *prefixDictionary) suggestFreq(term string, tk *Tokenizer) int {
	dSize := float64(pd.size)
	if dSize < 1.0 {
		dSize = 1.0
	}
	freq := 1.0
	pieces := tk.Cut(term, false)
	for _, p := range pieces {
		pieceFreq, found := pd.termFreq[p]
		if !found {
			pieceFreq = 1
		}
		freq *= float64(pieceFreq) / dSize
	}

	a := int(freq*dSize) + 1
	b := 1
	val, found := pd.termFreq[term]
	if found {
		b = val
	}
	if a > b {
		return a
	}
	return b
}

/*
buildDag(string) map[int][]int (dag)
findDagPath(string, dag) [][2]int (path)

*/

type hiddenMarkovModel struct {
	startP map[string]float64
	transP map[string]map[string]float64
	emitP  map[string]map[string]float64
	ready  bool
}

func newHMM(startProba map[string]float64, transitionProba, emitProba map[string]map[string]float64) hiddenMarkovModel {
	return hiddenMarkovModel{startProba, transitionProba, emitProba, true}
}

// Load jieba's trained Hidden Markov model.
func newJiebaHMM() hiddenMarkovModel {
	startP := map[string]float64{
		"B": -0.26268660809250016,
		"E": minFloat,
		"M": minFloat,
		"S": -1.4652633398537678,
	}
	transP := map[string]map[string]float64{
		"B": {
			"E": -0.51082562376599,  // B->E
			"M": -0.916290731874155, // B->M
		},
		"E": {
			"B": -0.5897149736854513, // E->B
			"S": -0.8085250474669937, // E->S
		},
		"M": {
			"E": -0.33344856811948514, // M->E
			"M": -1.2603623820268226,  // M->M
		},
		"S": {
			"B": -0.7211965654669841, // S->B
			"S": -0.6658631448798212, // S->S
		},
	}
	emitP := map[string]map[string]float64{} // "B": {"word": -1.1, ...}, ...
	jsonData, err := os.ReadFile("prob_emit.json")
	if err != nil {
		panic(fmt.Sprintf("failed to read prob_emit.json: %v", err))
	}
	err = json.Unmarshal(jsonData, &emitP)
	if err != nil {
		panic(fmt.Sprintf("failed to unmarshal json data: %v", err))
	}

	return newHMM(startP, transP, emitP)
}

// Use the Viterbi algorithm to find the hidden states of all
// characters in `text`, and the path of highest probability.
func (hmm *hiddenMarkovModel) viterbi(text string) []string {
	textRune := []rune(text)

	// Always return "S" for a single-piece input.
	if len(textRune) == 1 {
		return []string{"S"}
	}

	hiddenStateProba := map[int]map[string]float64{
		0: {},
	}
	fullPath := map[string][]string{
		"B": {"B"},
		"M": {"M"},
		"E": {"E"},
		"S": {"S"},
	}
	HMMstates := []string{"B", "M", "E", "S"}

	// Initial probabilities for each hidden state at rune[0]
	for _, s := range HMMstates {
		emit, found := hmm.emitP[s][string(textRune[0])]
		if !found {
			emit = minFloat
		}
		startProba := hmm.startP[s] + emit
		hiddenStateProba[0][s] = startProba
	}

	// Calculate probabilities for each hidden state from rune[1]
	// to rune[-1], and find the best route in between all state
	// transitions.
	for i_, char := range textRune[1:] {
		i := i_ + 1
		hiddenStateProba[i] = map[string]float64{}
		partialPath := map[string][]string{}
		// Find the most likely route preceding each state,
		// and the route's log probability.
		for _, s := range HMMstates {
			route := hmm.stateTransitionRoute(i, s, hiddenStateProba)
			emitProba, found := hmm.emitP[s][string(char)]
			if !found {
				emitProba = minFloat
			}
			stateProba := route.proba + emitProba
			hiddenStateProba[i][s] = stateProba
			// Append route to fullPath.
			partialPath[s] = append(partialPath[s], fullPath[route.from]...)
			partialPath[s] = append(partialPath[s], s)
		}
		fullPath = partialPath
	}

	// Select the path that arrives at either E or S state,
	// whichever has the highest hidden state probability.
	e := hiddenStateProba[len(textRune)-1]["E"]
	s := hiddenStateProba[len(textRune)-1]["S"]
	if e > s {
		return fullPath["E"]
	} else {
		return fullPath["S"]
	}
}

// Find the most likely route that connects one state to the next.
// For example, hidden state B could be preceded by either an E or
// a S. This function finds the most likely route (E->B vs S->B)
// along with the route's log probability.
func (hmm *hiddenMarkovModel) stateTransitionRoute(step int, nowState string, hiddenStates map[int]map[string]float64) transitionRoute {
	// List all possible routes and calculate their log probabilities.
	routes := map[string]float64{}
	for _, prevState := range stateChange[nowState] {
		prevProb := hiddenStates[step-1][prevState]
		routeProb := prevProb + hmm.transP[prevState][nowState]
		routes[prevState] = routeProb
	}

	// Pick the route with the highest log probability.
	bestPrevState := ""
	bestRouteProba := minFloat
	for prevState, routeProba := range routes {
		if routeProba > bestRouteProba {
			bestPrevState = prevState
			bestRouteProba = routeProba
		}
	}
	bestRoute := transitionRoute{bestPrevState, bestRouteProba}
	return bestRoute
}
