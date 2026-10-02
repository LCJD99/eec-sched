/* Pure session-trace parsing and projections.  The UI imports this module,
 * while Node's built-in test runner can exercise it without a DOM. */

export const KNOWN_EVENTS = [
  "run_started", "scheduler_loaded", "snapshot_loaded", "dataset_loaded",
  "evaluation_started", "evaluation_trace", "diagnosis_started", "session_start",
  "diagnosis_agent_started", "agent_context_prepared", "agent_started", "model_started",
  "round_start", "llm_request", "response_request", "llm_response", "model_output", "tool_call", "meta_tool_started",
  "meta_tool_completed", "tool_result", "tool_evaluation", "round_end", "model_failed",
  "agent_failed", "diagnosis_agent_failed", "session_end", "session_failed", "run_failed", "evaluation_completed",
];

const EVENT_SET = new Set(KNOWN_EVENTS);

export function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function asText(value) {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") return String(value);
  try { return JSON.stringify(value); } catch { return String(value); }
}

export function compactJson(value, max = 300) {
  const text = typeof value === "string" ? value : asText(value);
  return text.length <= max ? text : `${text.slice(0, Math.max(0, max - 1))}…`;
}

function numberValue(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value);
  return null;
}

export function eventType(record) {
  const value = record?.event_type ?? record?.event ?? record?.type;
  return value === undefined || value === null || value === "" ? "unknown" : String(value);
}

/**
 * Attach the active round and phase to every event without requiring traces to
 * repeat those fields on each JSONL line.  Events before the first round and
 * after a round_end stay outside a round; unannotated events inside a round
 * inherit the active round.
 */
export function deriveRoundContexts(records) {
  let round = null;
  let phase = null;
  return records.map((record) => {
    const nextRound = record?.round_number ?? record?.round;
    if (nextRound !== undefined && nextRound !== null) round = nextRound;
    if (record?.phase !== undefined && record.phase !== null) phase = record.phase;
    const context = { round, phase };
    if (eventType(record) === "round_end") {
      round = null;
      phase = null;
    }
    return context;
  });
}

/**
 * Partition the matching events into ordered visual groups.  The full record
 * sequence determines membership, so filtering cannot make inherited round
 * context disappear.
 */
export function groupRecordsByRound(records, matches = () => true) {
  const contexts = deriveRoundContexts(records);
  const groups = [];
  records.forEach((record, index) => {
    if (!matches(record)) return;
    const context = contexts[index];
    const previous = groups.at(-1);
    if (!previous || previous.round !== context.round) {
      groups.push({ round: context.round, phase: context.phase, events: [] });
    }
    groups.at(-1).events.push({ record, context, index });
  });
  return groups;
}

function sequenceValue(record) { return numberValue(record?.sequence); }

export function timestampValue(record) {
  const value = Date.parse(String(record?.timestamp ?? ""));
  return Number.isFinite(value) ? value : null;
}

export function parseJsonl(text) {
  const source = typeof text === "string" ? text : String(text ?? "");
  const records = [];
  const errors = [];
  const warnings = [];
  source.split(/\r?\n/).forEach((line, index) => {
    const lineNumber = index + 1;
    if (!line.trim()) return;
    let value;
    try { value = JSON.parse(line); } catch (error) {
      errors.push({ line: lineNumber, message: `Line ${lineNumber}: invalid JSON (${error instanceof Error ? error.message : "parse error"})` });
      return;
    }
    if (!isObject(value)) {
      errors.push({ line: lineNumber, message: `Line ${lineNumber}: JSON value must be an object` });
      return;
    }
    records.push({ ...value, __line: lineNumber });
  });

  const sessionIds = [...new Set(records.map((record) => record.session_id).filter(Boolean).map(String))];
  if (sessionIds.length > 1) warnings.push(`Mixed session_id values detected: ${sessionIds.join(", ")}`);
  const seenSequences = new Map();
  let previousSequence = null;
  let outOfOrder = false;
  let missingSequenceCount = 0;
  for (const record of records) {
    const sequence = sequenceValue(record);
    if (sequence === null) { missingSequenceCount += 1; continue; }
    if (seenSequences.has(sequence)) warnings.push(`Duplicate sequence ${sequence} on lines ${seenSequences.get(sequence)} and ${record.__line}`);
    else seenSequences.set(sequence, record.__line);
    if (previousSequence !== null && sequence < previousSequence) outOfOrder = true;
    previousSequence = sequence;
  }
  if (missingSequenceCount) warnings.push(`${missingSequenceCount} event${missingSequenceCount === 1 ? "" : "s"} missing a numeric sequence`);
  if (outOfOrder) warnings.push("Sequence values are out of order; events were sorted for display");
  const sorted = [...records].sort((left, right) => {
    const a = sequenceValue(left); const b = sequenceValue(right);
    if (a === null && b === null) return left.__line - right.__line;
    if (a === null) return 1; if (b === null) return -1;
    return a - b || left.__line - right.__line;
  });
  const ordered = [...seenSequences.keys()].filter((value) => Number.isInteger(value) && value > 0).sort((a, b) => a - b);
  const gaps = []; let expected = 1;
  for (const value of ordered) {
    if (value > expected) gaps.push(value - expected === 1 ? `${expected}` : `${expected}-${value - 1}`);
    expected = Math.max(expected, value + 1);
  }
  if (gaps.length) warnings.push(`Missing sequence value${gaps.length === 1 ? "" : "s"}: ${gaps.join(", ")}`);
  const unknownEvents = [...new Set(sorted.filter((record) => !EVENT_SET.has(eventType(record))).map((record) => eventType(record)))];
  if (unknownEvents.length) warnings.push(`Unknown event type${unknownEvents.length === 1 ? "" : "s"}: ${unknownEvents.join(", ")}`);
  return { records: sorted, errors, warnings, sessionIds, unknownEvents };
}

function valueAt(object, ...keys) {
  if (!isObject(object)) return undefined;
  for (const key of keys) if (object[key] !== undefined && object[key] !== null) return object[key];
  return undefined;
}

function nestedText(value, output = []) {
  if (typeof value === "string") { if (value) output.push(value); return output; }
  if (Array.isArray(value)) { value.forEach((item) => nestedText(item, output)); return output; }
  if (!isObject(value)) return output;
  for (const key of ["text", "reasoning_text", "output_text"]) {
    if (typeof value[key] === "string" && value[key]) output.push(value[key]);
  }
  if (value.content !== undefined) nestedText(value.content, output);
  if (value.summary !== undefined) nestedText(value.summary, output);
  return output;
}

function uniqueTexts(values) {
  const seen = new Set();
  return values.map((value) => String(value).trim()).filter((value) => value && !seen.has(value) && seen.add(value));
}

function outputEnvelope(record) {
  if (isObject(record?.output) && Array.isArray(record.output.output)) return record.output;
  if (isObject(record?.response)) return record.response;
  if (isObject(record?.output) && Array.isArray(record.output.items)) return record.output;
  if (Array.isArray(record?.output)) return { output: record.output };
  return isObject(record?.response) ? record.response : {};
}

function outputItems(record) {
  const envelope = outputEnvelope(record);
  return Array.isArray(envelope.output) ? envelope.output : Array.isArray(envelope.items) ? envelope.items : [];
}

function isFunctionCall(item) {
  return isObject(item) && (item.type === "function_call" || item.type === "tool_call" || isObject(item.function_call) || (isObject(item.function) && (item.name || item.function.name)));
}

function functionCallFields(item) {
  const nested = isObject(item?.function_call) ? item.function_call : isObject(item?.function) ? item.function : {};
  return {
    name: valueAt(item, "name") ?? valueAt(nested, "name"),
    callId: valueAt(item, "call_id", "callId") ?? valueAt(nested, "call_id", "callId") ?? valueAt(item, "id") ?? valueAt(nested, "id"),
    rawArguments: valueAt(item, "arguments", "args") ?? valueAt(nested, "arguments", "args"),
  };
}

export function decodeArguments(rawArguments) {
  if (isObject(rawArguments)) return { value: rawArguments, error: null };
  if (rawArguments === undefined || rawArguments === null || rawArguments === "") return { value: {}, error: null };
  if (typeof rawArguments !== "string") return { value: rawArguments, error: "Arguments are not a JSON object or JSON string" };
  try {
    const value = JSON.parse(rawArguments);
    if (!isObject(value)) return { value, error: "Decoded arguments are not an object" };
    return { value, error: null };
  } catch (error) {
    return { value: null, error: error instanceof Error ? error.message : "Invalid JSON arguments" };
  }
}

export function argumentSummary(value) {
  if (!isObject(value)) return compactJson(value, 240);
  const keys = Object.keys(value);
  return compactJson(value, 240) || (keys.length ? keys.join(", ") : "no arguments");
}

function usageProjection(usage) {
  if (!isObject(usage)) return null;
  const inputDetails = isObject(usage.input_tokens_details) ? usage.input_tokens_details : {};
  const outputDetails = isObject(usage.output_tokens_details) ? usage.output_tokens_details : {};
  return {
    input: numberValue(valueAt(usage, "input_tokens", "prompt_tokens")),
    output: numberValue(valueAt(usage, "output_tokens", "completion_tokens")),
    total: numberValue(valueAt(usage, "total_tokens")),
    reasoning: numberValue(valueAt(usage, "reasoning_tokens")) ?? numberValue(valueAt(outputDetails, "reasoning_tokens")),
    cached: numberValue(valueAt(usage, "cached_tokens", "cache_read_input_tokens")) ?? numberValue(valueAt(inputDetails, "cached_tokens")),
  };
}

export function projectModelOutput(record) {
  const envelope = outputEnvelope(record);
  const items = outputItems(record);
  const toolCalls = [];
  const reasoning = [];
  const messages = [];
  for (const item of items) {
    if (isFunctionCall(item)) {
      const fields = functionCallFields(item);
      const decoded = decodeArguments(fields.rawArguments);
      toolCalls.push({
        name: asText(fields.name) || "(unnamed tool)",
        callId: asText(fields.callId) || "(missing call_id)",
        rawArguments: fields.rawArguments,
        arguments: decoded.value,
        argumentsError: decoded.error,
        argumentSummary: decoded.error ? compactJson(fields.rawArguments, 240) : argumentSummary(decoded.value),
      });
      continue;
    }
    if (item?.type === "reasoning" || item?.type === "reasoning_text") reasoning.push(...nestedText(item.content ?? item.summary ?? item));
    else if (item?.type === "message" || item?.role === "assistant" || item?.type === "output_text") messages.push(...nestedText(item.content ?? item));
    else if (item?.content !== undefined) messages.push(...nestedText(item.content));
  }
  const fallbackText = valueAt(record, "output_text", "text");
  if (typeof fallbackText === "string") messages.push(fallbackText);
  const usage = usageProjection(valueAt(envelope, "usage"));
  return {
    toolCalls,
    reasoningText: uniqueTexts(reasoning).join("\n\n"),
    modelText: uniqueTexts(messages).join("\n\n"),
    metadata: {
      usage,
      responseId: valueAt(envelope, "response_id", "id"),
      requestId: valueAt(envelope, "request_id"),
      status: valueAt(record, "status") ?? valueAt(envelope, "status"),
    },
  };
}

export function projectResponseRequest(record) {
  const request = isObject(record?.input) ? record.input : isObject(record?.request) ? record.request : {};
  const input = Array.isArray(request.input) ? request.input : Array.isArray(request.messages) ? request.messages : Array.isArray(record?.input) ? record.input : [];
  const tools = Array.isArray(request.tools) ? request.tools : [];
  const toolNames = tools.map((tool) => isObject(tool?.function) ? tool.function.name : tool?.name).filter(Boolean).map(String);
  return { inputCount: input.length, toolNames, model: valueAt(request, "model") };
}

export function deriveMetrics(records) {
  const eventTypes = records.map(eventType);
  const modelOutputs = records.filter((record) => ["llm_response", "model_output"].includes(eventType(record)));
  const usage = modelOutputs.map((record) => projectModelOutput(record).metadata.usage).filter(Boolean);
  const times = records.map(timestampValue).filter((value) => value !== null);
  const firstTimestamp = times.length ? Math.min(...times) : null;
  const lastTimestamp = times.length ? Math.max(...times) : null;
  const rounds = new Set(records.map((record) => record.round_number ?? record.round).filter((value) => value !== undefined && value !== null)).size;
  const statuses = records.map((record) => record.status).filter(Boolean).map(String);
  let status = statuses.at(-1) ?? "unknown";
  if (eventTypes.some((value) => value.endsWith("_failed") || value === "run_failed")) status = "failed";
  return {
    events: records.length,
    rounds,
    modelRequests: eventTypes.filter((value) => value === "llm_request" || value === "response_request").length,
    modelOutputs: modelOutputs.length,
    toolCalls: eventTypes.filter((value) => value === "tool_call").length,
    toolResults: eventTypes.filter((value) => value === "tool_result").length,
    inputTokens: usage.reduce((sum, item) => sum + (item.input ?? 0), 0) || null,
    outputTokens: usage.reduce((sum, item) => sum + (item.output ?? 0), 0) || null,
    totalTokens: usage.reduce((sum, item) => sum + (item.total ?? 0), 0) || null,
    status,
    startTimestamp: firstTimestamp,
    endTimestamp: lastTimestamp,
    durationMs: firstTimestamp !== null && lastTimestamp !== null ? Math.max(0, lastTimestamp - firstTimestamp) : null,
  };
}

export function buildToolLedger(records) {
  const ledger = new Map();
  for (const record of records) {
    const type = eventType(record);
    if (!["tool_call", "tool_result", "tool_evaluation"].includes(type)) continue;
    const callId = record.call_id ? String(record.call_id) : `${record.tool ?? record.name ?? "tool"}:${record.sequence ?? record.__line}`;
    const current = ledger.get(callId) ?? { callId };
    if (type === "tool_call") Object.assign(current, { name: record.name ?? record.tool, arguments: record.arguments, requestedAt: record.timestamp, requestSequence: record.sequence });
    if (type === "tool_result") Object.assign(current, { result: record.result, resultStatus: record.status, completedAt: record.timestamp, resultSequence: record.sequence });
    if (type === "tool_evaluation") Object.assign(current, { evaluationStatus: record.status, executionOutcome: record.execution_outcome });
    ledger.set(callId, current);
  }
  return ledger;
}
