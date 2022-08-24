package tokenizer

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

type Tokenizer struct {
	CustomDict string
	initOk     bool
	prefixDict map[string]int
	dictSize   int
	startP     map[string]float64
	transP     map[string]map[string]float64
	emitP      map[string]map[string]float64
}

func (tk *Tokenizer) initialize() {
	// Build a prefix dictionary from user-proviced dictionary
	// file.
	if len(tk.CustomDict) != 0 {
		// Open & parse text file lines.
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

		// Build prefix dictionary
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
	prefixDict := map[string]int{}
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
			_, found := prefixDict[piece]
			if !found {
				prefixDict[piece] = 0
			}
		}
		prefixDict[word] = count
	}
	tk.prefixDict = prefixDict
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
	return dag
}

// Calculate the log probability of each DAG path (piece),
// and return the best path for each rune in `text`.
// The return value's index are based on []rune(text).
func findDAGPath(text string, dag map[int][]int, prefixDictionary map[string]int, dictSize int) [][2]int {
	total := math.Log(float64(dictSize))
	dagProba := map[int]map[int]float64{}

	// Iterate through `textRunes` in reverse.
	textRunes := []rune(text)
	for i := len(textRunes) - 1; i >= 0; i-- {
		dagProba[i] = map[int]float64{}
		for _, j := range dag[i] {
			// Calculate current piece's probability.
			// piece_frequency = log(prefix_dictionary.get(piece) or 1.0) - total
			// piece_proba = piece_frequency + next_piece_proba
			tf := 1.0
			if val, found := prefixDictionary[string(textRunes[i:j])]; found {
				tf = float64(val)
			}
			pieceFreq := math.Log(tf) - total

			// Get next piece's probability.
			nextPiece := map[int]float64{j: 0.0}
			if val, found := dagProba[j]; found {
				nextPiece = val
			}
			for _, nextPieceFreq := range nextPiece {
				pieceProba := pieceFreq + nextPieceFreq
				dagProba[i][j] = pieceProba
			}
			// fmt.Println(i, j)
		}
	}

	// Keep paths with the highest log probability.
	return findBestPath(text, dagProba)
}

// Find the path with the highest probability.
func findBestPath(text string, dagProba map[int]map[int]float64) [][2]int {
	textRunes := []rune(text)

	bestPath := [][2]int{}
	for i := 0; i < len(textRunes); {
		j := maxProbaIndex(dagProba[i])
		bestPath = append(bestPath, [2]int{i, j})
		i = j
	}
	return bestPath
}

// Return the map key whose value has the highest float.
func maxProbaIndex(probaIndex map[int]float64) int {
	bestIndex := -1
	bestProba := -3.14e100
	for i, proba := range probaIndex {
		if proba > bestProba {
			bestProba = proba
			bestIndex = i
		}
	}
	return bestIndex
}

func (tk *Tokenizer) loadHMM() {
	tk.startP = map[string]float64{
		"B": -0.26268660809250016,
		"E": -3.14e100,
		"M": -3.14e100,
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
