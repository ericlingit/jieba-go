package tokenizer

import (
	"encoding/gob"
	"fmt"
	"os"
	"reflect"
	"testing"
)

const dictSize = 60_101_967

var prefixDictionary = loadPrefixDictionaryFromGob()

func TestLoadHMM(t *testing.T) {
	tk := Tokenizer{}
	tk.loadHMM()
	if tk.emitP["B"]["一"] != -3.6544978750449433 {
		t.Error("load HMM failed")
	}
	if tk.emitP["M"]["一"] != -4.428158526435913 {
		t.Error("load HMM failed")
	}
	if tk.emitP["E"]["一"] != -6.044987536255073 {
		t.Error("load HMM failed")
	}
	if tk.emitP["S"]["一"] != -4.92368982120877 {
		t.Error("load HMM failed")
	}
}

func TestFindDAGPath(t *testing.T) {
	tk := Tokenizer{}
	tk.initOk = true
	tk.prefixDict = prefixDictionary
	tk.dictSize = dictSize

	t.Run("find DAG path: 今天天氣很好", func(t *testing.T) {
		text := "今天天氣很好"
		dag := map[int][]int{
			0: {1, 2}, // 今, 今天
			1: {2, 3}, // 天, 天天
			2: {3},    // 天
			3: {4},    // 氣
			4: {5},    // 很
			5: {6},    // 好
		}
		want := [][2]int{
			{0, 2}, // text[0:2] = 今天
			{2, 3}, // text[2:3] = 天
			{3, 4}, // text[3:4] = 氣
			{4, 5}, // text[4:5] = 很
			{5, 6}, // text[5:6] = 好
		}
		got := tk.findDAGPath(text, dag)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("want %v, got %v", want, got)
		}
	})

	t.Run("find DAG path: 我昨天去上海交通大學與老師討論量子力學", func(t *testing.T) {
		text := "我昨天去上海交通大學與老師討論量子力學"
		dag := map[int][]int{
			0:  {1},
			1:  {2, 3}, // 昨 昨天
			2:  {3},
			3:  {4},
			4:  {5, 6}, // 上 上海
			5:  {6},
			6:  {7},
			7:  {8},
			8:  {9},
			9:  {10},
			10: {11},
			11: {12},
			12: {13},
			13: {14},
			14: {15},
			15: {16, 17}, // 量 量子
			16: {17, 18}, // 子 子力
			17: {18},
			18: {19},
		}
		want := [][2]int{
			{0, 1},
			{1, 3},
			{3, 4},
			{4, 6},
			{6, 7},
			{7, 8},
			{8, 9},
			{9, 10},
			{10, 11},
			{11, 12},
			{12, 13},
			{13, 14},
			{14, 15},
			{15, 17},
			{17, 18},
			{18, 19},
		}
		got := tk.findDAGPath(text, dag)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("want %v, got %v", want, got)
		}
	})
}

func TestFindBestPath(t *testing.T) {
	t.Run("path1", func(t *testing.T) {
		dagProba := map[int]map[int]float64{
			5: {6: 1.1},         // 好
			4: {5: 1.1},         // 很
			3: {4: 1.1},         // 氣
			2: {3: 1.1},         // 天
			1: {2: 1.1, 3: 2.2}, // 天, 天天
			0: {1: 1.1, 2: 2.2}, // 今, 今天
		}
		want := [][2]int{
			{0, 2}, // 今天
			{2, 3},
			{3, 4},
			{4, 5},
			{5, 6},
		}
		text := "今天天氣很好"
		got := findBestPath(text, dagProba)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("want %v, got %v", want, got)
		}
	})

	t.Run("path2", func(t *testing.T) {
		dagProba := map[int]map[int]float64{
			18: {19: 1.1},
			17: {18: 1.1},
			16: {17: 1.1, 18: 2.2}, // 子, 子力
			15: {16: 1.1, 17: 2.2}, // 量, 量子
			14: {15: 1.1},
			13: {14: 1.1},
			12: {13: 1.1},
			11: {12: 1.1},
			10: {11: 1.1},
			9:  {10: 1.1},
			8:  {9: 1.1},
			7:  {8: 1.1},
			6:  {7: 1.1},
			5:  {6: 1.1},
			4:  {5: 1.1, 6: 2.2}, // 上, 上海
			3:  {4: 1.1},
			2:  {3: 1.1},
			1:  {2: 1.1, 3: 2.2}, // 昨, 昨天
			0:  {1: 1.1},
		}
		want := [][2]int{
			{0, 1},
			{1, 3}, // 昨天
			{3, 4},
			{4, 6}, // 上海
			{6, 7},
			{7, 8},
			{8, 9},
			{9, 10},
			{10, 11},
			{11, 12},
			{12, 13},
			{13, 14},
			{14, 15},
			{15, 17}, // 量子
			{17, 18},
			{18, 19},
		}
		text := "我昨天去上海交通大學與老師討論量子力學"
		got := findBestPath(text, dagProba)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("want %v, got %v", want, got)
		}
	})
}

func TestMaxProbaIndex(t *testing.T) {
	given := map[int]float64{
		0: 0.0,
		1: 1.1,
		2: 2.2,
		3: -3.3,
	}
	want := 2
	got := maxProbaIndex(given)
	if want != got {
		t.Errorf("want %v got %v", want, got)
	}
}

func TestBuildDAG(t *testing.T) {
	tk := Tokenizer{}
	tk.initOk = true
	tk.prefixDict = prefixDictionary
	tk.dictSize = dictSize

	text1 := "今天天氣很好"
	t.Run(fmt.Sprintf("DAG %s", text1), func(t *testing.T) {
		want := map[int][]int{
			0: {1, 2}, // text[0:1], text[0:2] == 今, 今天
			1: {2, 3}, // text[1:2], text[1:3] == 天, 天天
			2: {3},    // text[2:3] == 天
			3: {4},    // text[3:4] == 氣
			4: {5},    // text[4:5] == 很
			5: {6},    // text[5:6] == 好
		}
		got := tk.buildDAG(text1)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("want %v, got %v", want, got)
		}
	})

	text2 := "我昨天去上海交通大學與老師討論量子力學"
	t.Run(fmt.Sprintf("DAG %s", text2), func(t *testing.T) {
		want := map[int][]int{
			0:  {1},
			1:  {2, 3}, // 昨 昨天
			2:  {3},
			3:  {4},
			4:  {5, 6}, // 上 上海
			5:  {6},
			6:  {7, 8}, // 交 交通
			7:  {8},
			8:  {9},
			9:  {10},
			10: {11},
			11: {12},
			12: {13},
			13: {14},
			14: {15},
			15: {16, 17}, // 量 量子
			16: {17, 18}, // 子 子力
			17: {18},
			18: {19},
		}
		got := tk.buildDAG(text2)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("want %v, got %v", want, got)
		}
	})
}

func TestBuildPrefixDict(t *testing.T) {
	tk := Tokenizer{}
	input := []string{
		"AT&T 3 nz",
		"B超 3 n",
		"c# 3 nz",
		"C# 3",
		"江南style 3 n",
		"江南 4986 ns",
	}
	want := map[string]int{
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
	err := tk.buildPrefixDictionary(input)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, tk.prefixDict) {
		t.Errorf("want %v, got %v", want, tk.prefixDict)
	}
}

func TestInitialize(t *testing.T) {
	t.Run("with custom dictionary", func(t *testing.T) {
		f, _ := os.CreateTemp("", "aaa.txt")
		defer os.Remove(f.Name())
		f.Write([]byte("今天 10 x\n天氣 3\n"))

		tk := Tokenizer{}
		tk.CustomDict = f.Name()
		tk.initialize()
		if tk.initOk != true {
			t.Errorf("initialize failed")
		}
		want := map[string]int{
			"今":  0,
			"今天": 10,
			"天":  0,
			"天氣": 3,
		}
		if !reflect.DeepEqual(want, tk.prefixDict) {
			t.Errorf("want %v, got %v", want, tk.prefixDict)
		}
		wantSize := 13
		if wantSize != tk.dictSize {
			t.Errorf("want %v for dictSize, got %v", wantSize, tk.dictSize)
		}
	})

	t.Run("without custom dictionary", func(t *testing.T) {
		tk := Tokenizer{}
		tk.initialize()
		if tk.initOk != true {
			t.Errorf("initialize failed")
		}

		// Sample the first 100 items.
		i := 100
		for k, wantCount := range prefixDictionary {
			gotCount := tk.prefixDict[k]
			if wantCount != gotCount {
				t.Errorf("bad prefix dictionary. %q wants %d, got %d", k, wantCount, gotCount)
			}
			if i <= 0 {
				break
			}
			i--
		}
	})
}

// Load a prefix dictionary created from jieba's dict.txt.
func loadPrefixDictionaryFromGob() map[string]int {
	// Read gob file.
	gobFile, err := os.Open("prefix_dictionary.gob")
	if err != nil {
		panic(fmt.Sprintf("failed to open gob file: %v", err))
	}
	// Decode to pfDict map.
	pfDict := map[string]int{}
	decoder := gob.NewDecoder(gobFile)
	if err := decoder.Decode(&pfDict); err != nil {
		panic(fmt.Sprintf("failed to decode pfDict: %v", err))
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
