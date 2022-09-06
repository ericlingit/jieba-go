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
	"strconv"
	"strings"
	"sync"
)

const minFloat float64 = -3.14e100

type Tokenizer struct {
	CustomDict string
	initOk     bool
	prefixDict map[string]int
	dictSize   int
	startP     map[string]float64
	transP     map[string]map[string]float64
	emitP      map[string]map[string]float64
	// Values below are for debugging.
	dag      map[int][]int
	dagProba map[int]map[int]float64
}

// Initialize the Tokenizer. If CustomDict is specified, a prefix
// dictionary will be built from this file. If CustomDict is not
// specified, a pre-built prefix dictionary will be loaded.
func (tk *Tokenizer) initialize() {
	// Build a prefix dictionary from user-proviced dictionary
	// file.
	if len(tk.CustomDict) != 0 {
		// Open & collect dictionary file lines.
		reader, err := os.Open(tk.CustomDict)
		if err != nil {
			log.Fatalf("failed to read custom dictionary file: %v", err)
		}
		scanner := bufio.NewScanner(reader)
		scanner.Split(bufio.ScanLines)
		lines := []string{}
		for scanner.Scan() {
			lines = append(lines, scanner.Text())
		}
		reader.Close()

		// Build a prefix dictionary from lines collected during the above step.
		pdErr := tk.buildPrefixDictionary(lines)
		if pdErr != nil {
			log.Fatalf("failed to build prefix dictionary: %v", pdErr)
		}

		tk.initOk = true
		return
	}

	// Load pre-built prefix dictionary from gob file.
	gobFile, err := os.Open("prefix_dictionary.gob")
	if err != nil {
		log.Fatalf("failed to open gob file: %v", err)
	}
	// Decode to pfDict map.
	pfDict := map[string]int{}
	decoder := gob.NewDecoder(gobFile)
	if err := decoder.Decode(&pfDict); err != nil {
		log.Fatalf("failed to decode pfDict: %v", err)
	}
	tk.prefixDict = pfDict
	tk.initOk = true
}

type Line struct {
	text  string
	count int
	err   error
}

func splitLine(rawLine string) Line {
	parts := strings.SplitN(rawLine, " ", 3)
	word := parts[0]
	count, err := strconv.Atoi(parts[1])
	return Line{word, count, err}
}

func lineDigester(done chan struct{}, rawLineCh chan string, resultCh chan Line) {
	for rawLine := range rawLineCh {
		select {
		case <-done:
			return
		case resultCh <- splitLine(rawLine):
		}
	}
}

func (tk *Tokenizer) splitLineParallel(dictionaryLines []string) error {
	rawLineCh := make(chan string, len(dictionaryLines))
	for _, rawLine := range dictionaryLines {
		rawLineCh <- rawLine
	}
	close(rawLineCh)

	lineResultCh := make(chan Line, len(dictionaryLines))
	done := make(chan struct{})
	defer close(done)
	workerWg := sync.WaitGroup{}
	numWorkers := 6
	workerWg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go func() {
			lineDigester(done, rawLineCh, lineResultCh)
			workerWg.Done()
		}()
	}
	go func() {
		workerWg.Wait()
		close(lineResultCh)
	}()

	// Consume results.
	tk.prefixDict = make(map[string]int, len(dictionaryLines)*2)
	total := 0
	for _, line := range dictionaryLines {
		parts := strings.SplitN(line, " ", 3)
		word := parts[0]
		count, err := strconv.Atoi(parts[1])
		if err != nil {
			return err
		}
		total += count

		piece := ""
		for _, char := range word {
			piece += string(char)
			_, found := tk.prefixDict[piece]
			if !found {
				tk.prefixDict[piece] = 0
			}
		}
		tk.prefixDict[word] = count
	}
	tk.dictSize = total
	return nil
}

func produceLines(dictionaryLines []string) chan Line {
	c := make(chan Line)
	wg := sync.WaitGroup{}
	for _, dLine := range dictionaryLines {
		wg.Add(1)
		go func(l string) {
			defer wg.Done()
			c <- splitLine(l)
		}(dLine)
	}
	go func() {
		defer close(c)
		wg.Wait()
	}()
	return c
}

func consumeLines(lineIn chan Line, prefixDict map[string]int, dictLock *sync.Mutex, countCh chan int) chan struct{} {
	wg := sync.WaitGroup{}
	wg2 := sync.WaitGroup{}
	for _line := range lineIn {
		// Receive a Line from c channel, update prefixDict and
		wg.Add(1)
		go func(line Line) {
			// Update prefixDict.
			defer wg.Done()
			defer dictLock.Unlock()
			dictLock.Lock()
			piece := ""
			for _, char := range line.text {
				piece += string(char)
				_, found := prefixDict[piece]
				if !found {
					prefixDict[piece] = 0
				}
			}
			prefixDict[line.text] = line.count
		}(_line)

		// Update total (send count).
		wg2.Add(1)
		go func(l Line) {
			defer wg2.Done()
			countCh <- l.count
		}(_line)
	}

	go func() {
		wg2.Wait()
		close(countCh)
	}()
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	return done
}

func produceLines2(dictionaryLines []string) chan Line {
	c := make(chan Line)
	wg := sync.WaitGroup{}
	wg.Add(len(dictionaryLines))
	go func() {
		for _, dLine := range dictionaryLines {
			c <- splitLine(dLine)
			wg.Done()
		}
	}()
	go func() {
		wg.Wait()
		close(c)
	}()
	return c
}

func lineWorker(done chan struct{}, lineIn chan Line, prefixDict map[string]int, dictLock *sync.Mutex, countCh chan int) {
	for line := range lineIn {
		select {
		case <-done:
			return
		case countCh <- line.count:
			piece := ""
			dictLock.Lock()
			for _, char := range line.text {
				piece += string(char)
				_, found := prefixDict[piece]
				if !found {
					prefixDict[piece] = 0
				}
			}
			prefixDict[line.text] = line.count
			dictLock.Unlock()
		}
	}
}

func (tk *Tokenizer) buildPrefixDictionaryParallel2(dictionaryLines []string) {
	lock := sync.Mutex{}
	wg := sync.WaitGroup{}
	done := make(chan struct{})
	defer close(done)
	countCh := make(chan int)
	tk.prefixDict = make(map[string]int, len(dictionaryLines)*2)

	lineCh := produceLines2(dictionaryLines)
	numWorkers := 6
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go func() {
			lineWorker(done, lineCh, tk.prefixDict, &lock, countCh)
			wg.Done()
		}()
	}
	go func() {
		wg.Wait()
		close(countCh)
	}()
	for c := range countCh {
		tk.dictSize += c
	}
}

func (tk *Tokenizer) buildPrefixDictionaryParallel(dictionaryLines []string) {
	lock := sync.Mutex{}
	countCh := make(chan int)
	// tk.prefixDict = map[string]int{}
	tk.prefixDict = make(map[string]int, len(dictionaryLines)*2)

	lineCh := produceLines(dictionaryLines)
	done := consumeLines(lineCh, tk.prefixDict, &lock, countCh)
	for c := range countCh {
		tk.dictSize += c
	}
	<-done
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
	tk.prefixDict = make(map[string]int, len(dictionaryLines)*2)
	total := 0
	for _, line := range dictionaryLines {
		parts := strings.SplitN(line, " ", 3)
		word := parts[0]
		count, err := strconv.Atoi(parts[1])
		if err != nil {
			return err
		}
		total += count

		piece := ""
		for _, char := range word {
			piece += string(char)
			_, found := tk.prefixDict[piece]
			if !found {
				tk.prefixDict[piece] = 0
			}
		}
		tk.prefixDict[word] = count
	}
	tk.dictSize = total
	return nil
}

// Build a DAG out of every rune:rune+N piece from text string.
// The returned DAG's index values are based on []rune(text).
func (tk *Tokenizer) buildDAG(text string) map[int][]int {
	// Get the index of RUNES that are found in the prefix
	// dictionary. If not found, save the rune slice as is.
	textRunes := []rune(text)
	pieces := [][2]int{}
	for i, iRune := range textRunes {
		if _, found := tk.prefixDict[string(iRune)]; !found {
			pieces = append(pieces, [2]int{i, i + 1})
			continue
		}
		for j := range textRunes[i:] {
			part := textRunes[i : j+1+i]
			val, found := tk.prefixDict[string(part)]
			if !found {
				break
			}
			if val > 0 {
				pieces = append(pieces, [2]int{i, j + 1 + i})
			}
		}
	}

	dag := map[int][]int{}
	for _, p := range pieces {
		val, found := dag[p[0]]
		if !found {
			dag[p[0]] = []int{p[1]}
		} else {
			dag[p[0]] = append(val, p[1])
		}
	}
	tk.dag = dag
	return dag
}

// Calculate the log probability of each DAG path (piece),
// and return the best path for each rune in `text`.
// The return value's index are based on []rune(text).
func (tk *Tokenizer) findDAGPath(text string, dag map[int][]int) [][2]int {
	total := math.Log(float64(tk.dictSize))
	dagProba := map[int]map[int]float64{}

	// Iterate through `textRunes` in reverse.
	textRunes := []rune(text)
	for i := len(textRunes) - 1; i >= 0; i-- {
		// fmt.Printf("%q\n", string(textRunes[i]))
		dagProba[i] = map[int]float64{}
		for _, j := range dag[i] {
			// Calculate current piece's probability.
			// piece_frequency = log(prefix_dictionary.get(piece) or 1.0) - total
			// piece_proba = piece_frequency + next_piece_proba
			tf := 1.0
			if val, found := tk.prefixDict[string(textRunes[i:j])]; found {
				tf = float64(val)
			}
			pieceFreq := math.Log(tf) - total

			// Get next piece's probability.
			nextPiece := map[int]float64{j: 0.0}
			if val, found := dagProba[j]; found {
				nextPiece = val
			}
			// There could be more than 1 nextPiece, use the one
			// with the highest log probability.
			_, nextPieceBestProba := tk.maxIndexProba(nextPiece)
			pieceProba := pieceFreq + nextPieceBestProba
			dagProba[i][j] = pieceProba
			// fmt.Printf(
			// 	"  %q dagProba[%d][%d] = %f (%f %sFreq  + %f %sProba dagProba[%d][%d])\n",
			// 	string(textRunes[i:j]),
			// 	i,
			// 	j,
			// 	pieceProba,
			// 	pieceFreq,
			// 	string(textRunes[i:j]),
			// 	nextPieceBestProba,
			// 	string(textRunes[j:x]),
			// 	j,
			// 	x,
			// )
			// fmt.Println(i, j)
		}
	}
	tk.dagProba = dagProba
	// Keep paths with the highest log probability.
	return tk.findBestPath(text, dagProba)
}

// Find the path with the highest probability.
// This is a helper method for findDAGPath().
func (tk *Tokenizer) findBestPath(text string, dagProba map[int]map[int]float64) [][2]int {
	textRunes := []rune(text)

	bestPath := [][2]int{}
	for i := 0; i < len(textRunes); {
		j, _ := tk.maxIndexProba(dagProba[i])
		bestPath = append(bestPath, [2]int{i, j})
		i = j
	}
	return bestPath
}

// Return the map key whose float value is the highest.
func (tk *Tokenizer) maxIndexProba(probaIndex map[int]float64) (int, float64) {
	bestIndex := -1
	bestProba := minFloat
	for i, proba := range probaIndex {
		if proba > bestProba {
			bestProba = proba
			bestIndex = i
		}
	}
	return bestIndex, bestProba
}

// Load jieba's trained Hidden Markov model.
func (tk *Tokenizer) loadHMM() {
	tk.startP = map[string]float64{
		"B": -0.26268660809250016,
		"E": minFloat,
		"M": minFloat,
		"S": -1.4652633398537678,
	}
	tk.transP = map[string]map[string]float64{
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

	jsonData, err := os.ReadFile("prob_emit.json")
	if err != nil {
		panic(fmt.Sprintf("failed to read prob_emit.json: %v", err))
	}
	// tk.emitP := map[string]map[string]float64{} // "B": {"word": -1.1, ...}, ...
	err = json.Unmarshal(jsonData, &tk.emitP)
	if err != nil {
		panic(fmt.Sprintf("failed to unmarshal json data: %v", err))
	}
}

type transitionRoute struct {
	from  string
	proba float64
}

// Find the most likely route that connects one state to the next.
// For example, hidden state B could be preceded by either an E or
// a S. This function finds the most likely route (E->B vs S->B)
// along with the route's log probability.
func (tk *Tokenizer) stateTransitionRoute(step int, nowState string, hiddenStates map[int]map[string]float64) transitionRoute {
	stateChange := map[string][]string{
		"B": {"E", "S"}, // E->B, S->B
		"M": {"B", "M"},
		"E": {"B", "M"},
		"S": {"E", "S"},
	}

	// List all possible routes and their log probabilities.
	routes := map[string]float64{}
	for _, prevState := range stateChange[nowState] {
		prevProb := hiddenStates[step-1][prevState]
		routeProb := prevProb + tk.transP[prevState][nowState]
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

// Use the Viterbi algorithm to find the hidden states of all
// characters in `text`, and the path of highest probability.
func (tk *Tokenizer) viterbi(text string) []string {
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
		emit, found := tk.emitP[s][string(textRune[0])]
		if !found {
			emit = minFloat
		}
		startProba := tk.startP[s] + emit
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
			route := tk.stateTransitionRoute(i, s, hiddenStateProba)
			emitProba, found := tk.emitP[s][string(char)]
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
	finalState := "E"
	if e < s {
		finalState = "S"
	}
	return fullPath[finalState]
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

// Cut `text` using a prefix dictionary, and optionally use a
// Hidden Markov model to identify and segment unknown words.
func (tk *Tokenizer) Cut(text string, hmm bool) []string {
	dag := tk.buildDAG(text)
	dagPath := tk.findDAGPath(text, dag)
	dagPieces := tk.cutDAG(text, dagPath)
	if !hmm {
		return dagPieces
	}

	// Use HMM to segment uncut chars in dagPieces.
	tk.loadHMM()
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
				v := tk.viterbi(string(uncutRunes))
				newWords := tk.cutHMM(string(uncutRunes), v)
				words = append(words, newWords...)
				uncutRunes = nil
			}
		} else {
			// Run cutHMM when a length > 1 rune is encountered.
			if len(uncutRunes) != 0 {
				v := tk.viterbi(string(uncutRunes))
				newWords := tk.cutHMM(string(uncutRunes), v)
				words = append(words, newWords...)
				uncutRunes = nil
			}
			words = append(words, piece)
		}
	}

	return words
}

var hanzi = regexp.MustCompile(`\p{Han}+`)

type TextBlock struct {
	text  string
	doCut bool
}

// Identify which blocks to perform segmentation, and which
// ones to skip by splitting `text` into a slice of Hanzi
// and non-Hanzi blocks. Hanzi TextBlocks will have their
// `doCut` value set to `true`.
func textSplitter(text string) []TextBlock {
	zhIndexes := hanzi.FindAllIndex([]byte(text), -1)
	if len(zhIndexes) == 0 {
		return []TextBlock{{text, false}}
	}
	// FindAllIndex() returns a slice of index slices.
	// Example:
	//     [][]int{
	//         {4, 6},
	//         {8, 10},
	//     }
	// Each slice of indexes marks the start (inclusive),
	// and the end (exclusive) of matched Hanzi *bytes*.
	//
	// The for-loop below finds all the non-Hanzi indexes.
	// For example, if `text` has a length of 15 bytes, then
	// we'll use the following indexes to split `text`:
	//     [][]int{
	//         {0, 4},
	//         {4, 6}, (hanzi index bounds)
	//         {6, 8},
	//         {8, 10}, (hanzi index bounds)
	//         {10, 15},
	//     }

	blocks := []TextBlock{}
	// Check left side of each zhIndex[0].
	for i, bound := range zhIndexes {
		if i == 0 && bound[0] != 0 {
			blocks = append(blocks, TextBlock{text[0:bound[0]], false})
		}
		if i != 0 && bound[0] != zhIndexes[i-1][1] {
			blocks = append(blocks, TextBlock{text[zhIndexes[i-1][1]:bound[0]], false})
		}
		blocks = append(blocks, TextBlock{text[bound[0]:bound[1]], true})
		// Check right side of last zhIndex.
		if i == len(zhIndexes)-1 {
			if bound[1] < len(text) {
				blocks = append(blocks, TextBlock{text[bound[1]:], false})
			}
		}
	}
	return blocks
}
