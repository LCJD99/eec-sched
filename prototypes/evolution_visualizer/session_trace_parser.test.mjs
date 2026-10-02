import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { decodeArguments, groupRecordsByRound, parseJsonl, projectModelOutput } from "./session_trace_parser.mjs";

const line = (value) => JSON.stringify(value);

test("projects the current llm_response envelope and keeps function calls before text", () => {
  const record = {
    event_type: "llm_response",
    output: {
      output: [
        { type: "reasoning", content: [{ type: "reasoning_text", text: "Inspect the trace first." }] },
        { type: "function_call", call_id: "call-good", name: "profile_trace", arguments: '{"trace_id":"mnms-000319"}' },
        { type: "function_call", call_id: "call-bad", name: "validate_intervention", arguments: "{not-json" },
        { type: "message", content: [{ type: "output_text", text: "I will inspect the primary DAG." }] },
      ],
      usage: { input_tokens: 11, output_tokens: 7, total_tokens: 18, output_tokens_details: { reasoning_tokens: 3 } },
    },
  };
  const projection = projectModelOutput(record);
  assert.equal(projection.toolCalls.length, 2);
  assert.equal(projection.toolCalls[0].name, "profile_trace");
  assert.deepEqual(projection.toolCalls[0].arguments, { trace_id: "mnms-000319" });
  assert.equal(projection.toolCalls[0].argumentsError, null);
  assert.match(projection.toolCalls[1].rawArguments, /not-json/);
  assert.ok(projection.toolCalls[1].argumentsError);
  assert.equal(projection.reasoningText, "Inspect the trace first.");
  assert.equal(projection.modelText, "I will inspect the primary DAG.");
  assert.deepEqual(projection.metadata.usage, { input: 11, output: 7, total: 18, reasoning: 3, cached: null });
});

test("accepts an object argument and reports non-object JSON without dropping the raw value", () => {
  assert.deepEqual(decodeArguments({ section: "nodes" }), { value: { section: "nodes" }, error: null });
  const result = decodeArguments("[1,2,3]");
  assert.deepEqual(result.value, [1, 2, 3]);
  assert.match(result.error, /not an object/);
});

test("keeps unknown events and valid lines around malformed JSON", () => {
  const parsed = parseJsonl([
    line({ sequence: 1, event_type: "session_start" }),
    line({ sequence: 2, event_type: "brand_new_event", payload: { safe: true } }),
    "{broken",
    line({ sequence: 3, event_type: "llm_response", output: { output: [] } }),
  ].join("\n"));
  assert.equal(parsed.records.length, 3);
  assert.equal(parsed.errors.length, 1);
  assert.deepEqual(parsed.unknownEvents, ["brand_new_event"]);
  assert.equal(parsed.records.find((record) => record.event_type === "brand_new_event").payload.safe, true);
});

test("groups filtered events by inherited round context", () => {
  const records = [
    { sequence: 1, event_type: "session_start" },
    { sequence: 2, event_type: "round_start", round_number: 1, phase: "diagnosis" },
    { sequence: 3, event_type: "tool_call" },
    { sequence: 4, event_type: "round_end", round_number: 1 },
    { sequence: 5, event_type: "tool_call" },
    { sequence: 6, event_type: "round_start", round_number: 2, phase: "diagnosis" },
    { sequence: 7, event_type: "tool_call" },
  ];
  const groups = groupRecordsByRound(records, (record) => record.event_type === "tool_call" || record.event_type === "round_end");
  assert.deepEqual(groups.map((group) => [group.round, group.phase, group.events.map((event) => event.record.sequence)]), [
    [1, "diagnosis", [3, 4]],
    [null, null, [5]],
    [2, "diagnosis", [7]],
  ]);
});

test("projects the checked-in 130-event trace when present", (context) => {
  const filename = "outputs/bottleneck-analysis/20260921T132234569831Z.jsonl";
  if (!fs.existsSync(filename)) { context.skip("fixture unavailable"); return; }
  const parsed = parseJsonl(fs.readFileSync(filename, "utf8"));
  assert.equal(parsed.records.length, 130);
  const responses = parsed.records.filter((record) => record.event_type === "llm_response");
  assert.equal(responses.length, 10);
  assert.ok(responses.some((record) => projectModelOutput(record).toolCalls.length > 0));
});
