package tokenizer

import (
	"encoding/gob"
	"fmt"
	"os"
	"reflect"
	"testing"
)

var prefixDictionary = loadPrefixDictionaryFromGob()

func TestBuildDAG(t *testing.T) {
	text := "今天天氣很好"
	want := map[int][]int{
		0: {1, 2}, // text[0:1], text[0:2] == 今, 今天
		1: {2, 3}, // text[1:2], text[1:3] == 天, 天天
		2: {3},    // text[2:3] == 天
		3: {4},    // text[3:4] == 氣
		4: {5},    // text[4:5] == 很
		5: {6},    // text[5:6] == 好
	}
	got := buildDAG(text, prefixDictionary)
	if !reflect.DeepEqual(want, got) {
		t.Errorf("want %v, got %v", want, got)
	}
}

func TestBuildPrefixDict(t *testing.T) {
	input := []string{
		"AT&T 3 nz",
		"B超 3 n",
		"c# 3 nz",
		"C# 3",
		"江南style 3 n",
		"江南 4986 ns",
	}
	want := map[string]uint{
		"A":       0,
		"AT":      0,
		"AT&":     0,
		"AT&T":    3,
		"B":       0,
		"B超":      3,
		"c":       0,
		"c#":      3,
		"C#":      3,
		"C":       0,
		"江":       0,
		"江南":      4986,
		"江南s":     0,
		"江南st":    0,
		"江南sty":   0,
		"江南styl":  0,
		"江南style": 3,
	}
	got, err := buildPrefixDictionary(input)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("want %v, got %v", want, got)
	}
}

// Load a prefix dictionary created from jieba's dict.txt.
func loadPrefixDictionaryFromGob() map[string]uint {
	// Read gob file.
	gobFile, err := os.Open("prefix_dictionary.gob")
	if err != nil {
		panic(fmt.Sprintf("failed to create a gob file: %v", err))
	}
	// Decode to pfDict map.
	pfDict := map[string]uint{}
	decoder := gob.NewDecoder(gobFile)
	if err := decoder.Decode(&pfDict); err != nil {
		panic(fmt.Sprintf("failed to encode pfDict: %v", err))
	}

	return pfDict
}

// func savePrefixDictionaryToGob() {
// 	// Read dict.txt.
// 	reader, err := os.Open("dict.txt")
// 	if err != nil {
// 		panic(fmt.Sprintf("failed to read dict.txt: %v", err))
// 	}
// 	scanner := bufio.NewScanner(reader)
// 	scanner.Split(bufio.ScanLines)

// 	lines := []string{}
// 	for scanner.Scan() {
// 		lines = append(lines, scanner.Text())
// 	}
// 	reader.Close()

// 	// Build prefix dictionary.
// 	pfDict, err := buildPrefixDictionary(lines)
// 	if err != nil {
// 		panic(fmt.Sprintf("failed to build prefix dictionary: %v", err))
// 	}

// 	// Serialize pfDict map to a gob.
// 	gobFile, err := os.Create("prefix_dictionary.gob")
// 	if err != nil {
// 		panic(fmt.Sprintf("failed to create a gob file: %v", err))
// 	}
// 	encoder := gob.NewEncoder(gobFile)
// 	if err := encoder.Encode(pfDict); err != nil {
// 		panic(fmt.Sprintf("failed to encode pfDict: %v", err))
// 	}
// }
