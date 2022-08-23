package tokenizer

import (
	"math"
	"strconv"
	"strings"
)

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
func buildPrefixDictionary(dictionaryLines []string) (map[string]uint, error) {
	prefixDict := map[string]uint{}
	for _, line := range dictionaryLines {
		parts := strings.SplitN(line, " ", 3)
		word := parts[0]
		count, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, err
		}

		piece := ""
		for _, char := range word {
			piece += string(char)
			_, found := prefixDict[piece]
			if !found {
				prefixDict[piece] = 0
			}
		}
		prefixDict[word] = uint(count)
	}
	return prefixDict, nil
}

// Build a DAG out of every rune:rune+N piece from text string.
// The returned DAG's index values are based on []rune(text).
func buildDAG(text string, prefixDictionary map[string]uint) map[int][]int {
	// Get the index of RUNES that are found in the prefix
	// dictionary. If not found, save the rune slice as is.
	textRunes := []rune(text)
	pieces := [][2]int{}
	for i, iRune := range textRunes {
		if _, found := prefixDictionary[string(iRune)]; !found {
			pieces = append(pieces, [2]int{i, i + 1})
			continue
		}
		for j := range textRunes[i:] {
			part := textRunes[i : j+1+i]
			val, found := prefixDictionary[string(part)]
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
func findDAGPath(text string, dag map[int][]int, prefixDictionary map[string]uint, dictSize int) [][2]int {
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
	bestPath := [][2]int{}
	bestJ := 0
	for i := 0; i < len(textRunes); i = bestJ {
		bestProba := math.SmallestNonzeroFloat64
		for j, proba := range dagProba[i] {
			if proba > bestProba {
				bestProba = proba
				bestJ = j
			}
			bestJ = j
		}
		bestPath = append(bestPath, [2]int{i, bestJ})
		// fmt.Println(i, bestJ)
	}
	return bestPath
}
