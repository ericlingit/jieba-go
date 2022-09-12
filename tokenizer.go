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
	"unicode"
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

		wordR := []rune(word)
		piece := ""
		for _, char := range wordR[:len(wordR)-1] {
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

// cutText `text` using a prefix dictionary, and optionally use a
// Hidden Markov model to identify and segment unknown words.
func (tk *Tokenizer) cutText(text string, hmm bool) []string {
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

func (tk *Tokenizer) Cut(text string, hmm bool) []string {
	zhIndexes := hanzi.FindAllIndex([]byte(text), -1)
	blocks := textSplitter(text, zhIndexes)

	result := []string{}
	for _, block := range blocks {
		if block.doProcess {
			result = append(result, tk.cutText(block.text, hmm)...)
		} else {
			result = append(result, processNonZh(block.text)...)
		}
	}
	return result
}

var hanzi = regexp.MustCompile(`\p{Han}+`)

type TextBlock struct {
	text      string
	doProcess bool
}

// Identify the text index ranges to process.
func textSplitter(text string, markedIndexes [][]int) []TextBlock {
	if len(markedIndexes) == 0 {
		return []TextBlock{{text, false}}
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
	blocks := []TextBlock{}
	prevTail := 0
	for i, pair := range markedIndexes {
		// Pair has left-side gap.
		if pair[0] != prevTail {
			// Fill in the gap.
			filler := text[prevTail:pair[0]]
			blocks = append(blocks, TextBlock{filler, false})
		}
		markedText := text[pair[0]:pair[1]]
		blocks = append(blocks, TextBlock{markedText, true})
		prevTail = pair[1]

		// Last pair with a right-side gap.
		if i == len(markedIndexes)-1 && pair[1] != len(text) {
			// Fill in the gap.
			filler := text[pair[1]:]
			blocks = append(blocks, TextBlock{filler, false})
		}
	}
	return blocks
}

var alnum = regexp.MustCompile(`([a-zA-Z0-9]+)`)

// Perform simple segmentation for space delimited alphanumeric
// words. All other characters are broken into individual runes.
func processNonZh(text string) []string {
	alnumIdx := alnum.FindAllIndex([]byte(text), -1)
	if len(alnumIdx) == 0 {
		return []string{""}
	}

	textPieces := []string{}
	blocks := textSplitter(text, alnumIdx)
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
