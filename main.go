package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/openai/openai-go/v3/shared"
	myrlm "rlm/gorlm"
)

type OolongExample struct {
	Dataset           string `json:"dataset"`
	Question          string `json:"question"`
	Answer            string `json:"answer"`
	ContextWindowText string `json:"context_window_text"`
}

// OOLONG-Pairs queries from Appendix D.1 of the RLM paper (arXiv:2512.24601v2).
var oolongPairsQueries = []string{
	// Task 1
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or entity. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 2
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or human being. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 3
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 4
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a human being or location, and all instances that are a human being for both users must be after January 6, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 5
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or numeric value, and all instances that are an entity for both users must be before March 15, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 6
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a location or abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 7
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or numeric value, and all instances that are a numeric value for both users must be after February 1, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 8
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a human being or description and abstract concept. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 9
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or location, and all instances that are a location for both users must be after April 10, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 10
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a numeric value or abbreviation, and all instances that are an abbreviation for both users must be before May 20, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 11
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity and one with abbreviation, and the other user has exactly one instance with entity. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 12
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with numeric value, and the other user has at least one instance with location and at least one instance with human being. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 13
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with description and abstract concept, and the other user has at least one instance with abbreviation and at least one instance with entity. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 14
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with human being and at least one instance with numeric value, and the other user has exactly two instances with location. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 15
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity, at least one instance with location, and at least one instance with abbreviation, and the other user has exactly one instance with numeric value. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 16
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with description and abstract concept and at least one instance with human being, and the other user has at least two instances with entity and exactly one instance with abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 17
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with numeric value, and the other user has at least one instance with location and at least one instance with description and abstract concept. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 18
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with abbreviation and exactly one instance with human being, and the other user has at least one instance with entity and at least one instance with numeric value. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 19
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with location and at least one instance with entity, and the other user has exactly one instance with description and abstract concept and exactly one instance with abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
	// Task 20
	`In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with numeric value and at least one instance with human being, and the other user has at least one instance with location, at least one instance with entity, and exactly one instance with abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.`,
}

func loadOolongExample(path string) (*OolongExample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read example file: %w", err)
	}
	var ex OolongExample
	if err := json.Unmarshal(data, &ex); err != nil {
		return nil, fmt.Errorf("parse example JSON: %w", err)
	}
	return &ex, nil
}

const (
	oolongModel             = "gpt-5"
	defaultOolongDatasetKey = "65k"
)

type oolongDatasetSpec struct {
	Path string
}

var oolongDatasets = map[string]oolongDatasetSpec{
	"65k": {
		Path: "rlm/examples/oolong_trec_coarse_example.json",
	},
	"16k": {
		Path: "data/oolong/oolong_trec_coarse_16k.json",
	},
}

// hardTasks are the 5 tasks most likely to trip up the RLM:
//   - asymmetric constraints ("one user has X, the other has Y")
//   - exact-count requirements ("exactly one", "exactly two")
//   - date filters combined with the above
// Tasks 11, 14, 16, 19, 20 (1-indexed).
var hardTasks = []int{11, 14, 16, 19, 20}

func parseTasks(spec string) ([]int, error) {
	parts := strings.Split(spec, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n < 1 || n > len(oolongPairsQueries) {
			return nil, fmt.Errorf("invalid task number %q (must be 1-%d)", p, len(oolongPairsQueries))
		}
		out = append(out, n)
	}
	return out, nil
}

func resolveDataset(key string) (string, error) {
	spec, ok := oolongDatasets[key]
	if !ok {
		return "", fmt.Errorf("invalid --dataset %q (supported: 16k, 65k)", key)
	}

	if _, err := os.Stat(spec.Path); err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("dataset file %q not found", spec.Path)
		}
		return "", fmt.Errorf("stat dataset file %q: %w", spec.Path, err)
	}

	return spec.Path, nil
}

func main() {
	tasksFlag := flag.String("tasks", "", "comma-separated task numbers to run (1-20), e.g. '11,14,16'")
	itersFlag := flag.Int("iters", 10, "max RLM iterations per task")
	datasetFlag := flag.String("dataset", defaultOolongDatasetKey, "dataset size preset to use: 16k or 65k")
	flag.Parse()

	taskNums := hardTasks
	if *tasksFlag != "" {
		var err error
		taskNums, err = parseTasks(*tasksFlag)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	dataPath, err := resolveDataset(strings.TrimSpace(strings.ToLower(*datasetFlag)))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	example, err := loadOolongExample(dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load data: %v\n", err)
		os.Exit(1)
	}

	effort := shared.ReasoningEffortLow
	client := myrlm.NewOpenAIClient(myrlm.ClientConfig{
		Model:        myrlm.SharedModel(oolongModel),
		ReasoningEff: &effort,
	})

	rlm := myrlm.NewRLM(client,
		myrlm.WithMaxIterations(*itersFlag),
		myrlm.WithDockerConfig(myrlm.DockerConfig{
			Model: oolongModel,
		}),
	)

	contextText := example.ContextWindowText

	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Printf("  OOLONG-Pairs Benchmark — tasks %v — model: %s\n", taskNums, oolongModel)
	fmt.Printf("  Context: %d chars (%s dataset) — max iters: %d\n", len(contextText), example.Dataset, *itersFlag)
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	totalStart := time.Now()

	for idx, taskNum := range taskNums {
		query := oolongPairsQueries[taskNum-1]
		fmt.Printf("── Task %d (query #%d) [%d/%d] ──\n", taskNum, taskNum, idx+1, len(taskNums))
		fmt.Printf("Query: %.120s...\n\n", query)

		rlmCtx := myrlm.Context{
			Content: contextText,
			Metadata: myrlm.ContextMetadata{
				Type:   "string",
				Length: len(contextText),
			},
		}

		taskStart := time.Now()
		answer, err := rlm.Completion(context.Background(), rlmCtx, myrlm.Query(query))
		taskDuration := time.Since(taskStart)

		if err != nil {
			fmt.Fprintf(os.Stderr, "\n[ERROR] Task %d: %v\n\n", taskNum, err)
			continue
		}

		fmt.Println("Answer:")
		fmt.Println(answer)
		fmt.Printf("\nTask time: %s\n\n", taskDuration.Round(time.Millisecond))
	}

	fmt.Println(client.Usage.Summary())
	fmt.Printf("  Wall clock:        %s\n", time.Since(totalStart).Round(time.Millisecond))
}
