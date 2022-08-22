package tokenizer

import (
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
