package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

type label string

const (
	labelDesc  label = "description and abstract concept"
	labelEnty  label = "entity"
	labelHum   label = "human being"
	labelNum   label = "numeric value"
	labelLoc   label = "location"
	labelAbbr  label = "abbreviation"
	dateLayout       = "Jan 02, 2006"
)

type labeledEntry struct {
	UserID int
	Date   time.Time
	Label  label
}

type userStats struct {
	Counts        map[label]int
	LatestByLabel map[label]time.Time
}

type pair struct {
	A int
	B int
}

func newPair(a, b int) pair {
	if a < b {
		return pair{A: a, B: b}
	}
	return pair{A: b, B: a}
}

func (u userStats) count(l label) int {
	return u.Counts[l]
}

func (u userStats) hasAny(labels ...label) bool {
	for _, l := range labels {
		if u.Counts[l] > 0 {
			return true
		}
	}
	return false
}

func (u userStats) allBefore(l label, t time.Time) bool {
	if u.Counts[l] == 0 {
		return true
	}
	return u.LatestByLabel[l].Before(t)
}

func (u userStats) allAfter(l label, t time.Time) bool {
	if u.Counts[l] == 0 {
		return true
	}
	return u.LatestByLabel[l].After(t)
}

func main() {
	labelsPath := flag.String("labels", "", "Path to labeled entries JSON/JSONL (required)")
	predsDir := flag.String("preds_dir", "", "Directory with prediction files named task_<n>.txt")
	goldDir := flag.String("gold_out_dir", "", "If set, writes gold pairs as task_<n>.txt")
	tasksFlag := flag.String("tasks", "1-20", "Tasks to evaluate, e.g. '1-20' or '11,14,16'")
	flag.Parse()

	if *labelsPath == "" {
		exitf("missing --labels")
	}
	if *predsDir == "" && *goldDir == "" {
		exitf("set at least one of --preds_dir or --gold_out_dir")
	}

	taskIDs, err := parseTasks(*tasksFlag)
	if err != nil {
		exitf("invalid --tasks: %v", err)
	}

	entries, err := loadLabeledEntries(*labelsPath)
	if err != nil {
		exitf("load labels: %v", err)
	}
	stats := buildUserStats(entries)

	if len(stats) < 2 {
		exitf("need at least 2 users in labels file")
	}

	if *goldDir != "" {
		if err := os.MkdirAll(*goldDir, 0o755); err != nil {
			exitf("create gold_out_dir: %v", err)
		}
	}

	fmt.Printf("Loaded %d entries for %d users\n", len(entries), len(stats))
	fmt.Printf("Evaluating tasks: %v\n\n", taskIDs)

	var macroF1 float64
	var taskCount int
	for _, taskID := range taskIDs {
		gold := computeGoldPairs(taskID, stats)
		if *goldDir != "" {
			outPath := filepath.Join(*goldDir, fmt.Sprintf("task_%d.txt", taskID))
			if err := writePairs(outPath, gold); err != nil {
				exitf("write gold for task %d: %v", taskID, err)
			}
		}

		if *predsDir == "" {
			fmt.Printf("Task %2d  gold=%6d\n", taskID, len(gold))
			continue
		}

		predPath := filepath.Join(*predsDir, fmt.Sprintf("task_%d.txt", taskID))
		pred, err := loadPredPairs(predPath)
		if err != nil {
			exitf("load prediction for task %d: %v", taskID, err)
		}
		p, r, f1 := scoreF1(gold, pred)
		fmt.Printf("Task %2d  gold=%6d pred=%6d  P=%.4f R=%.4f F1=%.4f\n", taskID, len(gold), len(pred), p, r, f1)
		macroF1 += f1
		taskCount++
	}

	if taskCount > 0 {
		fmt.Printf("\nMacro F1 over %d tasks: %.4f\n", taskCount, macroF1/float64(taskCount))
	}
}

func loadLabeledEntries(path string) ([]labeledEntry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return nil, fmt.Errorf("empty labels file")
	}

	// Supports JSON array or JSONL for convenience.
	if strings.HasPrefix(trimmed, "[") {
		var raw []map[string]any
		if err := json.Unmarshal(data, &raw); err != nil {
			return nil, err
		}
		out := make([]labeledEntry, 0, len(raw))
		for i, row := range raw {
			entry, err := parseLabeledRow(row)
			if err != nil {
				return nil, fmt.Errorf("row %d: %w", i, err)
			}
			out = append(out, entry)
		}
		return out, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []labeledEntry
	sc := bufio.NewScanner(f)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		entry, err := parseLabeledRow(row)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		out = append(out, entry)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func parseLabeledRow(row map[string]any) (labeledEntry, error) {
	userID, err := getInt(row, "user_id", "userId", "user")
	if err != nil {
		return labeledEntry{}, fmt.Errorf("missing user id: %w", err)
	}
	rawLabel, err := getString(row, "label", "coarse_label", "category")
	if err != nil {
		return labeledEntry{}, fmt.Errorf("missing label: %w", err)
	}
	rawDate, err := getString(row, "date", "timestamp")
	if err != nil {
		return labeledEntry{}, fmt.Errorf("missing date: %w", err)
	}
	parsedLabel, err := normalizeLabel(rawLabel)
	if err != nil {
		return labeledEntry{}, err
	}
	parsedDate, err := parseDate(rawDate)
	if err != nil {
		return labeledEntry{}, fmt.Errorf("bad date %q: %w", rawDate, err)
	}
	return labeledEntry{
		UserID: userID,
		Date:   parsedDate,
		Label:  parsedLabel,
	}, nil
}

func parseDate(s string) (time.Time, error) {
	layouts := []string{
		dateLayout,
		"2006-01-02",
		time.RFC3339,
		time.RFC3339Nano,
	}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unsupported format")
}

func normalizeLabel(v string) (label, error) {
	s := strings.ToLower(strings.TrimSpace(v))
	switch s {
	case "description and abstract concept", "desc", "description":
		return labelDesc, nil
	case "entity", "enty":
		return labelEnty, nil
	case "human being", "hum":
		return labelHum, nil
	case "numeric value", "num":
		return labelNum, nil
	case "location", "loc":
		return labelLoc, nil
	case "abbreviation", "abbr":
		return labelAbbr, nil
	default:
		return "", fmt.Errorf("unknown label %q", v)
	}
}

func buildUserStats(entries []labeledEntry) map[int]userStats {
	out := make(map[int]userStats)
	for _, e := range entries {
		u, ok := out[e.UserID]
		if !ok {
			u = userStats{
				Counts:        make(map[label]int),
				LatestByLabel: make(map[label]time.Time),
			}
		}
		u.Counts[e.Label]++
		if e.Date.After(u.LatestByLabel[e.Label]) {
			u.LatestByLabel[e.Label] = e.Date
		}
		out[e.UserID] = u
	}
	return out
}

func computeGoldPairs(taskID int, stats map[int]userStats) map[pair]struct{} {
	var userIDs []int
	for id := range stats {
		userIDs = append(userIDs, id)
	}
	sort.Ints(userIDs)

	mustAfterJan6, _ := time.Parse(dateLayout, "January 06, 2023")
	mustBeforeMar15, _ := time.Parse(dateLayout, "March 15, 2023")
	mustAfterFeb1, _ := time.Parse(dateLayout, "February 01, 2023")
	mustAfterApr10, _ := time.Parse(dateLayout, "April 10, 2023")
	mustBeforeMay20, _ := time.Parse(dateLayout, "May 20, 2023")

	pairs := make(map[pair]struct{})

	addIf := func(uID, vID int, ok bool) {
		if ok {
			pairs[newPair(uID, vID)] = struct{}{}
		}
	}

	asym := func(u, v userStats, left, right func(userStats) bool) bool {
		return (left(u) && right(v)) || (left(v) && right(u))
	}

	for i := 0; i < len(userIDs); i++ {
		uID := userIDs[i]
		u := stats[uID]
		for j := i + 1; j < len(userIDs); j++ {
			vID := userIDs[j]
			v := stats[vID]

			switch taskID {
			case 1:
				addIf(uID, vID, u.hasAny(labelDesc, labelEnty) && v.hasAny(labelDesc, labelEnty))
			case 2:
				addIf(uID, vID, u.hasAny(labelEnty, labelHum) && v.hasAny(labelEnty, labelHum))
			case 3:
				addIf(uID, vID, u.hasAny(labelDesc, labelAbbr) && v.hasAny(labelDesc, labelAbbr))
			case 4:
				addIf(uID, vID,
					u.hasAny(labelHum, labelLoc) &&
						v.hasAny(labelHum, labelLoc) &&
						u.allAfter(labelHum, mustAfterJan6) &&
						v.allAfter(labelHum, mustAfterJan6))
			case 5:
				addIf(uID, vID,
					u.hasAny(labelEnty, labelNum) &&
						v.hasAny(labelEnty, labelNum) &&
						u.allBefore(labelEnty, mustBeforeMar15) &&
						v.allBefore(labelEnty, mustBeforeMar15))
			case 6:
				addIf(uID, vID, u.hasAny(labelLoc, labelAbbr) && v.hasAny(labelLoc, labelAbbr))
			case 7:
				addIf(uID, vID,
					u.hasAny(labelDesc, labelNum) &&
						v.hasAny(labelDesc, labelNum) &&
						u.allAfter(labelNum, mustAfterFeb1) &&
						v.allAfter(labelNum, mustAfterFeb1))
			case 8:
				addIf(uID, vID, u.hasAny(labelHum, labelDesc) && v.hasAny(labelHum, labelDesc))
			case 9:
				addIf(uID, vID,
					u.hasAny(labelEnty, labelLoc) &&
						v.hasAny(labelEnty, labelLoc) &&
						u.allAfter(labelLoc, mustAfterApr10) &&
						v.allAfter(labelLoc, mustAfterApr10))
			case 10:
				addIf(uID, vID,
					u.hasAny(labelNum, labelAbbr) &&
						v.hasAny(labelNum, labelAbbr) &&
						u.allBefore(labelAbbr, mustBeforeMay20) &&
						v.allBefore(labelAbbr, mustBeforeMay20))
			case 11:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelEnty) >= 1 && s.count(labelAbbr) >= 1 },
					func(s userStats) bool { return s.count(labelEnty) == 1 },
				))
			case 12:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelNum) >= 2 },
					func(s userStats) bool { return s.count(labelLoc) >= 1 && s.count(labelHum) >= 1 },
				))
			case 13:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelDesc) == 1 },
					func(s userStats) bool { return s.count(labelAbbr) >= 1 && s.count(labelEnty) >= 1 },
				))
			case 14:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelHum) >= 1 && s.count(labelNum) >= 1 },
					func(s userStats) bool { return s.count(labelLoc) == 2 },
				))
			case 15:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelEnty) >= 1 && s.count(labelLoc) >= 1 && s.count(labelAbbr) >= 1 },
					func(s userStats) bool { return s.count(labelNum) == 1 },
				))
			case 16:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelDesc) >= 1 && s.count(labelHum) >= 1 },
					func(s userStats) bool { return s.count(labelEnty) >= 2 && s.count(labelAbbr) == 1 },
				))
			case 17:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelNum) == 1 },
					func(s userStats) bool { return s.count(labelLoc) >= 1 && s.count(labelDesc) >= 1 },
				))
			case 18:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelAbbr) >= 1 && s.count(labelHum) == 1 },
					func(s userStats) bool { return s.count(labelEnty) >= 1 && s.count(labelNum) >= 1 },
				))
			case 19:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelLoc) >= 2 && s.count(labelEnty) >= 1 },
					func(s userStats) bool { return s.count(labelDesc) == 1 && s.count(labelAbbr) == 1 },
				))
			case 20:
				addIf(uID, vID, asym(u, v,
					func(s userStats) bool { return s.count(labelNum) >= 1 && s.count(labelHum) >= 1 },
					func(s userStats) bool { return s.count(labelLoc) >= 1 && s.count(labelEnty) >= 1 && s.count(labelAbbr) == 1 },
				))
			}
		}
	}
	return pairs
}

func loadPredPairs(path string) (map[pair]struct{}, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parsePairs(string(data)), nil
}

func writePairs(path string, pairs map[pair]struct{}) error {
	list := make([]pair, 0, len(pairs))
	for p := range pairs {
		list = append(list, p)
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].A == list[j].A {
			return list[i].B < list[j].B
		}
		return list[i].A < list[j].A
	})

	var b strings.Builder
	for i, p := range list {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(fmt.Sprintf("(%d, %d)", p.A, p.B))
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func parsePairs(s string) map[pair]struct{} {
	re := regexp.MustCompile(`\(\s*(\d+)\s*,\s*(\d+)\s*\)`)
	matches := re.FindAllStringSubmatch(s, -1)
	out := make(map[pair]struct{}, len(matches))
	for _, m := range matches {
		a, _ := strconv.Atoi(m[1])
		b, _ := strconv.Atoi(m[2])
		if a == b {
			continue
		}
		out[newPair(a, b)] = struct{}{}
	}
	return out
}

func scoreF1(gold, pred map[pair]struct{}) (precision, recall, f1 float64) {
	if len(pred) == 0 && len(gold) == 0 {
		return 1, 1, 1
	}
	var tp int
	for p := range pred {
		if _, ok := gold[p]; ok {
			tp++
		}
	}
	if len(pred) > 0 {
		precision = float64(tp) / float64(len(pred))
	}
	if len(gold) > 0 {
		recall = float64(tp) / float64(len(gold))
	}
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}
	return precision, recall, f1
}

func parseTasks(spec string) ([]int, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil, fmt.Errorf("empty")
	}
	if spec == "1-20" {
		out := make([]int, 20)
		for i := 1; i <= 20; i++ {
			out[i-1] = i
		}
		return out, nil
	}
	parts := strings.Split(spec, ",")
	out := make([]int, 0, len(parts))
	seen := make(map[int]struct{})
	for _, part := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || n < 1 || n > 20 {
			return nil, fmt.Errorf("bad task id %q", part)
		}
		if _, ok := seen[n]; ok {
			continue
		}
		seen[n] = struct{}{}
		out = append(out, n)
	}
	sort.Ints(out)
	return out, nil
}

func getString(row map[string]any, keys ...string) (string, error) {
	for _, k := range keys {
		if v, ok := row[k]; ok {
			switch t := v.(type) {
			case string:
				return t, nil
			}
		}
	}
	return "", fmt.Errorf("none of keys %v", keys)
}

func getInt(row map[string]any, keys ...string) (int, error) {
	for _, k := range keys {
		if v, ok := row[k]; ok {
			switch t := v.(type) {
			case float64:
				return int(t), nil
			case int:
				return t, nil
			case string:
				n, err := strconv.Atoi(strings.TrimSpace(t))
				if err == nil {
					return n, nil
				}
			}
		}
	}
	return 0, fmt.Errorf("none of keys %v", keys)
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
