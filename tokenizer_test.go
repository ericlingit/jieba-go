package tokenizer

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"reflect"
	"testing"
)

const dictSize = 60_101_967

var prefixDictionary = loadPrefixDictionaryFromGob()

func TestCutBigTextParallel(t *testing.T) {
	tk := newTokenizer(true)
	text := loadBigText()
	tk.CutParallel(text, true, 6, false)
}

func TestCutBigText(t *testing.T) {
	tk := newTokenizer(true)
	text := loadBigText()
	tk.Cut(text, true)
}

func TestCut(t *testing.T) {
	tk := newTokenizer(true)
	cases := []struct {
		name string
		text string
		want []string
		hmm  bool
	}{
		{"cut 1", "今天天氣很好", []string{"今天", "天", "氣", "很", "好"}, false},
		{"cut 1 hmm", "今天天氣很好", []string{"今天", "天氣", "很", "好"}, true},
		{"cut 2", "我昨天去上海交通大學與老師討論量子力學", []string{"我", "昨天", "去", "上海", "交通", "大", "學", "與", "老", "師", "討", "論", "量子", "力", "學"}, false},
		{"cut 2 hmm", "我昨天去上海交通大學與老師討論量子力學", []string{"我", "昨天", "去", "上海", "交通", "大學", "與", "老師", "討論", "量子", "力學"}, true},
		{"cut 3 hmm", "english번역『하다』今天天氣很好，ステーションabc1231+1=2我昨天去上海*important*去", []string{"english", "번", "역", "『", "하", "다", "』", "今天", "天氣", "很", "好", "，", "ス", "テ", "ー", "シ", "ョ", "ン", "abc1231", "+", "1", "=", "2", "我", "昨天", "去", "上海", "*", "important", "*", "去"}, true},
		{"cut 4", "some english words", []string{"some", "english", "words"}, false},
		{"cut 5", "abc123", []string{"abc123"}, false},
		{"cut 6", "a1+1=2", []string{"a1", "+", "1", "=", "2"}, false},
		{"cut 7", "aaa\nbbb", []string{"aaa", "bbb"}, false},
		{"cut 8", "这一刹那的撙近", []string{"这", "一刹那", "的", "撙", "近"}, false},
		{"cut 9", "这一刹那的撙近", []string{"这", "一刹那", "的", "撙近"}, true},
		{"cut 10", "撙", []string{"撙"}, false}, // This character causes memory leak.
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := tk.Cut(c.text, c.hmm)
			if !reflect.DeepEqual(c.want, got) {
				t.Logf("tk.dag: %v", tk.dag)
				t.Logf("tk.dagProba: %v", tk.dagProba)
				t.Fatalf("%q wants %v, got %v", c.name, c.want, got)
			}
		})
	}
}

func TestSplitText(t *testing.T) {
	cases := []struct {
		text string
		want []textBlock
	}{
		{"xxx中文xxx", []textBlock{{0, "xxx", false}, {1, "中文", true}, {2, "xxx", false}}},
		{"中文xxx", []textBlock{{0, "中文", true}, {1, "xxx", false}}},
		{"xxx中文", []textBlock{{0, "xxx", false}, {1, "中文", true}}},
		{"xxx", []textBlock{{0, "xxx", false}}},
		{"中文", []textBlock{{0, "中文", true}}},
		{"english번역『하다』今天天氣很好，ステーション1+1=2我昨天去上海*important*去", []textBlock{{0, "english번역『하다』", false}, {1, "今天天氣很好", true}, {2, "，ステーション1+1=2", false}, {3, "我昨天去上海", true}, {4, "*important*", false}, {5, "去", true}}},
	}
	for _, c := range cases {
		t.Run(c.text, func(t *testing.T) {
			zhIndexes := zh.FindAllIndex([]byte(c.text), -1)
			got := splitText(c.text, zhIndexes)
			assertDeepEqual(t, c.want, got)
		})
	}
}

func TestBuildDAG(t *testing.T) {
	tk := newTokenizer(false)
	cases := []struct {
		text string
		want map[int][]int
	}{
		{"今天天氣很好", map[int][]int{
			0: {1, 2}, // text[0:1], text[0:2] == 今, 今天
			1: {2, 3}, // text[1:2], text[1:3] == 天, 天天
			2: {3},    // text[2:3] == 天
			3: {4},    // text[3:4] == 氣
			4: {5},    // text[4:5] == 很
			5: {6},    // text[5:6] == 好
		}},
		{"我昨天去上海交通大學與老師討論量子力學", map[int][]int{
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
		}},
		{"这一刹那的撙近", map[int][]int{
			0: {1},
			1: {2, 3, 4}, // 一 一刹 一刹那
			2: {3, 4},    // 刹 刹那
			3: {4},
			4: {5},
			5: {6},
			6: {7},
		}},
		{"撙", map[int][]int{0: {1}}}, // "撙" is in prefix dict; has count == 0.
	}
	for _, c := range cases {
		t.Run(c.text, func(t *testing.T) {
			got := tk.buildDAG(c.text)
			assertDeepEqual(t, c.want, got)
		})
	}
}

func TestFindDAGPath(t *testing.T) {
	tk := newTokenizer(false)
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
		assertDeepEqual(t, want, got)
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
		got := tk.findDAGPath(text, dag)
		assertDeepEqual(t, want, got)
	})
}

func TestMaxIndexProba(t *testing.T) {
	tk := Tokenizer{}
	cases := []struct {
		candidates []tailProba
		wantIdx    int
		wantProba  float64
	}{
		{
			[]tailProba{
				{0, 0.0},
				{1, 1.1},
				{2, 2.2},
				{3, -3.3},
			},
			2,
			2.2,
		},
		{
			[]tailProba{
				{5, -3.14e100},
			},
			5,
			-3.14e100,
		},
		{
			[]tailProba{
				{2, -3.14e100},
				{3, -3.14e100},
				{4, -3.14e100},
			},
			4,
			-3.14e100,
		},
	}
	for i, c := range cases {
		t.Run(fmt.Sprintf("case %d", i), func(t *testing.T) {
			got := tk.maxIndexProba(c.candidates)
			assertEqual(t, c.wantIdx, got.Index)
			assertEqual(t, c.wantProba, got.Proba)
		})
	}
}

func TestFindBestPath(t *testing.T) {
	tk := Tokenizer{}
	cases := []struct {
		text     string
		dagProba map[int][]tailProba
		want     [][2]int
	}{
		{
			"今天天氣很好",
			map[int][]tailProba{
				5: {{6, 1.1}},           // 好
				4: {{5, 1.1}},           // 很
				3: {{4, 1.1}},           // 氣
				2: {{3, 1.1}},           // 天
				1: {{2, 1.1}, {3, 2.2}}, // 天, 天天
				0: {{1, 1.1}, {2, 2.2}}, // 今, 今天
			},
			[][2]int{
				{0, 2}, // 今天
				{2, 3},
				{3, 4},
				{4, 5},
				{5, 6},
			},
		},
		{
			"我昨天去上海交通大學與老師討論量子力學",
			map[int][]tailProba{
				18: {{19, 1.1}},
				17: {{18, 1.1}},
				16: {{17, 1.1}, {18, 2.2}}, // 子, 子力
				15: {{16, 1.1}, {17, 2.2}}, // 量, 量子
				14: {{15, 1.1}},
				13: {{14, 1.1}},
				12: {{13, 1.1}},
				11: {{12, 1.1}},
				10: {{11, 1.1}},
				9:  {{10, 1.1}},
				8:  {{9, 1.1}},
				7:  {{8, 1.1}},
				6:  {{7, 1.1}},
				5:  {{6, 1.1}},
				4:  {{6, 2.2}, {5, 1.1}}, // 上海, 上
				3:  {{4, 1.1}},
				2:  {{3, 1.1}},
				1:  {{2, 1.1}, {3, 2.2}}, // 昨, 昨天
				0:  {{1, 1.1}},
			},
			[][2]int{
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
			},
		},
		{
			"这一刹那的撙近",
			map[int][]tailProba{
				6: {{7, 1.1}},                     // 近
				5: {{6, 1.1}},                     // 撙
				4: {{5, 1.1}},                     // 的
				3: {{4, 1.1}},                     // 那
				2: {{3, 1.1}, {4, 2.2}},           // 刹 刹那
				1: {{2, 1.1}, {3, 2.2}, {4, 3.3}}, // 一 一刹 一刹那
				0: {{1, 1.1}},                     // 这
			},
			[][2]int{
				{0, 1}, // 这
				{1, 4}, // 一刹那
				{4, 5}, // 的
				{5, 6}, // 撙
				{6, 7}, // 近
			},
		},
	}
	for _, c := range cases {
		t.Run(c.text, func(t *testing.T) {
			got := tk.findBestPath(c.text, c.dagProba)
			assertDeepEqual(t, c.want, got)
		})
	}
}

func TestCutDag(t *testing.T) {
	tk := newTokenizer(false)
	t.Run("cut dag 1", func(t *testing.T) {
		text := "今天天氣很好"
		dPath := [][2]int{
			{0, 2},
			{2, 3},
			{3, 4},
			{4, 5},
			{5, 6},
		}
		want := []string{"今天", "天", "氣", "很", "好"}
		got := tk.cutDAG(text, dPath)
		assertDeepEqual(t, want, got)
	})

	t.Run("cut dag 2", func(t *testing.T) {
		text := "我昨天去上海交通大學與老師討論量子力學"
		want := []string{"我", "昨天", "去", "上海", "交通", "大", "學", "與", "老", "師", "討", "論", "量子", "力", "學"}
		dPath := [][2]int{
			{0, 1},
			{1, 3}, // 昨天
			{3, 4},
			{4, 6}, // 上海
			{6, 8}, // 交通
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
		got := tk.cutDAG(text, dPath)
		assertDeepEqual(t, want, got)
	})
}

func TestLoadHMM(t *testing.T) {
	tk := newTokenizer(false)
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

func TestViterbi(t *testing.T) {
	tk := newTokenizer(true)
	t.Run("viterbi case 1", func(t *testing.T) {
		text := "天氣很好"
		want := []string{"B", "E", "S", "S"}
		got := tk.viterbi(text)
		assertDeepEqual(t, want, got)
	})

	t.Run("viterbi case 2", func(t *testing.T) {
		text := "大學與老師討論"
		want := []string{"B", "E", "S", "B", "E", "B", "E"}
		got := tk.viterbi(text)
		assertDeepEqual(t, want, got)
	})
}

func TestStateTransitionRoute(t *testing.T) {
	tk := newTokenizer(true)
	hsProb := map[int]map[string]float64{
		0: {"B": 1.1, "M": 1.1, "E": 1.1, "S": 1.1},
		1: {"B": 1.1, "M": 1.1, "E": 1.1, "S": 1.1},
	}
	step := 2
	cases := []struct {
		name     string
		wantFrom string
		nowState string
	}{
		{"transition E->B vs S->B", "E", "B"},
		{"transition B->M vs M->M", "B", "M"},
		{"transition B->E vs M->E", "M", "E"},
		{"transition B->S vs M->S", "S", "S"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gotRoute := tk.stateTransitionRoute(step, c.nowState, hsProb)
			assertEqual(t, c.wantFrom, gotRoute.from)
		})
	}
}

func TestCutHMM(t *testing.T) {
	tk := newTokenizer(true)
	t.Run("cut hmm 1", func(t *testing.T) {
		text := "天氣很好"
		vPath := []string{"B", "E", "S", "S"}
		want := []string{"天氣", "很", "好"}
		got := tk.cutHMM(text, vPath)
		assertDeepEqual(t, want, got)

	})

	t.Run("cut hmm 2", func(t *testing.T) {
		text := "大學與老師討論"
		vPath := []string{"B", "E", "S", "B", "E", "B", "E"}
		want := []string{"大學", "與", "老師", "討論"}
		got := tk.cutHMM(text, vPath)
		assertDeepEqual(t, want, got)
	})
}

func TestCutNonZh(t *testing.T) {
	tk := Tokenizer{}
	cases := []struct {
		text string
		want []string
	}{
		{"some english words", []string{"some", "english", "words"}},
		{"abc123", []string{"abc123"}},
		{"a1+1=2", []string{"a1", "+", "1", "=", "2"}},
		{"aaa\nbbb", []string{"aaa", "bbb"}},
	}
	for _, c := range cases {
		got := tk.cutNonZh(c.text)
		if !reflect.DeepEqual(c.want, got) {
			t.Errorf("case %q: want %v, got %v", c.text, c.want, got)
		}
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
		assertDeepEqual(t, want, tk.prefixDict)
		wantSize := 13
		assertEqual(t, wantSize, tk.dictSize)
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
	assertDeepEqual(t, want, tk.prefixDict)
}

func TestBuildPrefixDictFromScratch(t *testing.T) {
	tk := Tokenizer{}
	tk.CustomDict = "dict.txt"
	lines := loadDictionaryFile(tk.CustomDict)

	tk.buildPrefixDictionary(lines)
	// Compare ALL items in `prefixDictionary` to
	// `tk.prefixDict`.
	assertDeepEqualLoop(t, prefixDictionary, tk.prefixDict)
}

func TestAddWord(t *testing.T) {
	tk := Tokenizer{}
	tk.prefixDict = map[string]int{}
	newWords := map[string]int{
		"左和右": 20,
		"上和下": 80,
	}
	for word, freq := range newWords {
		tk.AddWord(word, freq)
	}
	for word, freq := range newWords {
		val, found := tk.prefixDict[word]
		if !found {
			t.Errorf("want %q in prefixDict, not found", word)
		}
		if val != freq {
			t.Errorf("want %d for %q, got %d", freq, word, val)
		}
	}
	if tk.dictSize != 100 {
		t.Errorf("want 100 for dictSize, got %d", tk.dictSize)
	}
}

//
// Benchmarks.
//

// 92,710,594 ns/op
func BenchmarkCutBigTextParallel(b *testing.B) {
	tk := newTokenizer(true)
	text := loadBigText()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.CutParallel(text, true, 6, false)
	}
}

// 318,559,415 ns/op
func BenchmarkCutBigText(b *testing.B) {
	tk := newTokenizer(true)
	text := loadBigText()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.Cut(text, true)
	}
}

// 42,705 ns/op
func BenchmarkCut(b *testing.B) {
	tk := newTokenizer(true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.Cut("我昨天去上海交通大學與老師討論量子力學", true)
	}
}

// 4,4289 ns/op
func BenchmarkBuildDag(b *testing.B) {
	tk := newTokenizer(true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.buildDAG("我昨天去上海交通大學與老師討論量子力學")
	}
}

// 9,598 ns/op
func BenchmarkFindDAGPath(b *testing.B) {
	tk := newTokenizer(false)
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.findDAGPath("我昨天去上海交通大學與老師討論量子力學", dag)
	}
}

// 1,140 ns/op
func BenchmarkFindBestPath(b *testing.B) {
	tk := Tokenizer{}
	dagProba := map[int][]tailProba{
		18: {{19, 1.1}},
		17: {{18, 1.1}},
		16: {{17, 1.1}, {18, 2.2}}, // 子, 子力
		15: {{16, 1.1}, {17, 2.2}}, // 量, 量子
		14: {{15, 1.1}},
		13: {{14, 1.1}},
		12: {{13, 1.1}},
		11: {{12, 1.1}},
		10: {{11, 1.1}},
		9:  {{10, 1.1}},
		8:  {{9, 1.1}},
		7:  {{8, 1.1}},
		6:  {{7, 1.1}},
		5:  {{6, 1.1}},
		4:  {{6, 2.2}, {5, 1.1}}, // 上海, 上
		3:  {{4, 1.1}},
		2:  {{3, 1.1}},
		1:  {{2, 1.1}, {3, 2.2}}, // 昨, 昨天
		0:  {{1, 1.1}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.findBestPath("我昨天去上海交通大學與老師討論量子力學", dagProba)
	}
}

// 1,039 ns/op
func BenchmarkCutDag(b *testing.B) {
	tk := newTokenizer(false)
	dag := [][2]int{
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.cutDAG("我昨天去上海交通大學與老師討論量子力學", dag)
	}
}

// 64,731 ns/op
func BenchmarkViterbi(b *testing.B) {
	tk := newTokenizer(true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.viterbi("我昨天去上海交通大學與老師討論量子力學")
	}
}

// 140,353,760 ns/op
func BenchmarkBuildPrefDict(b *testing.B) {
	tk := Tokenizer{}
	tk.CustomDict = "dict.txt"
	lines := loadDictionaryFile(tk.CustomDict)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tk.buildPrefixDictionary(lines)
	}
}

/*
go test -bench=. -benchmem
goos: linux
goarch: amd64
pkg: github.com/ericlingit/jieba-go
cpu: Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz
BenchmarkCutBigTextParallel-6                 12          92710594 ns/op        165219404 B/op   2331807 allocs/op
BenchmarkCutBigText-6                          4         327280788 ns/op        134408870 B/op   2331841 allocs/op
BenchmarkCut-6                             31472             36895 ns/op           19716 B/op        322 allocs/op
BenchmarkBuildDag-6                       251918              4237 ns/op            2473 B/op         32 allocs/op
BenchmarkFindDAGPath-6                    243849              4607 ns/op            2161 B/op         30 allocs/op
BenchmarkFindBestPath-6                  2248059               528 ns/op             496 B/op          5 allocs/op
BenchmarkCutDag-6                        1000000              1064 ns/op             624 B/op         21 allocs/op
BenchmarkViterbi-6                         17935             66896 ns/op           52982 B/op        508 allocs/op
BenchmarkBuildPrefDict-6                       7         149018897 ns/op        51680593 B/op    1346011 allocs/op
*/

func newTokenizer(hmm bool) Tokenizer {
	tk := Tokenizer{}
	tk.initOk = true
	tk.dictSize = dictSize
	tk.prefixDict = prefixDictionary
	if hmm {
		tk.loadHMM()
	}
	// The return statement copies a sync.RWMutex lock.
	// This is intentional. Each Tokenizer instance in
	// every test should be independent of each other.
	return tk
}

func assertDeepEqual(t *testing.T, want, got interface{}) {
	t.Helper()
	if !reflect.DeepEqual(want, got) {
		t.Errorf("want %v, got %v", want, got)
	}
}

func assertEqual(t *testing.T, want, got interface{}) {
	t.Helper()
	if want != got {
		t.Errorf("want %v, got %v", want, got)
	}
}

// Use a for-loop to perform reflect.DeepEqual. This is
// much faster than calling DeepEqual.
func assertDeepEqualLoop(t *testing.T, want, got map[string]int) {
	t.Helper()
	for k, v := range want {
		if v != got[k] {
			t.Errorf("%q want %v, got %v", k, v, got[k])
		}
	}
}

func loadBigText() string {
	data, err := os.ReadFile("围城.txt")
	if err != nil {
		log.Fatal("failed to open 围城.txt", err)
	}
	return string(data)
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

func loadDictionaryFile(f string) []string {
	reader, err := os.Open(f)
	if err != nil {
		panic(fmt.Sprintf("failed to read custom dictionary file: %v\n", err))
	}
	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanLines)
	lines := []string{}
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	reader.Close()
	return lines
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

// func TestAAA(t *testing.T) {
// 	cases := []struct {
// 		indexes [][]int
// 		length  int
// 		want    [][]int
// 	}{
// 		{[][]int{{0, 6}, {8, 10}}, 15, [][]int{{0, 6}, {6, 8}, {8, 10}, {10, 15}}},
// 		{[][]int{{4, 6}, {8, 10}}, 15, [][]int{{0, 4}, {4, 6}, {6, 8}, {8, 10}, {10, 15}}},
// 		{[][]int{{4, 6}, {8, 15}}, 15, [][]int{{0, 4}, {4, 6}, {6, 8}, {8, 15}}},
// 		{[][]int{{4, 8}}, 15, [][]int{{0, 4}, {4, 8}, {8, 15}}},
// 		{[][]int{{0, 15}}, 15, [][]int{{0, 15}}},
// 	}
// 	for _, c := range cases {
// 		got := fillIndexGaps(c.indexes, c.length)
// 		if !reflect.DeepEqual(c.want, got) {
// 			t.Errorf("want %v, got %v", c.want, got)
// 		}
// 	}
// }
