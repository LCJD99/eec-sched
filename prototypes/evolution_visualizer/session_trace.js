import {
  asText,
  buildToolLedger,
  compactJson,
  deriveMetrics,
  eventType,
  groupRecordsByRound,
  parseJsonl,
  projectModelOutput,
  projectResponseRequest,
} from "./session_trace_parser.mjs";

const $ = (selector) => document.querySelector(selector);
const state = { traces: [], activeFilename: null, parsed: null, ledger: new Map(), traceText: "" };

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function append(parent, ...children) {
  children.filter(Boolean).forEach((child) => parent.append(child));
  return parent;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString(undefined, { maximumFractionDigits: digits }) : String(value);
}

function formatTime(value) {
  if (!value) return "time unavailable";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value) : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "medium" });
}

function formatDuration(milliseconds) {
  if (milliseconds === null || milliseconds === undefined) return "—";
  if (milliseconds < 1000) return `${formatNumber(milliseconds, 1)} ms`;
  return `${formatNumber(milliseconds / 1000, 2)} s`;
}

function jsonText(value, pretty = true) {
  try { return JSON.stringify(value, null, pretty ? 2 : 0); } catch { return String(value); }
}

const JSON_TOKEN_RE = /"(?:\\["\\/bfnrt]|\\u[0-9a-fA-F]{4}|[^"\\\u0000-\u001F])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|[{}\[\],:]/y;

function jsonTokenClass(token, source, end) {
  if (token[0] === '"') {
    let next = end;
    while (/\s/.test(source[next] || "")) next += 1;
    return source[next] === ":" ? "json-key" : "json-string";
  }
  if (token === "true" || token === "false") return "json-boolean";
  if (token === "null") return "json-null";
  if (/^-?(?:\d|\.\d)/.test(token)) return "json-number";
  return "json-punctuation";
}

function appendJsonTokens(parent, source) {
  let offset = 0;
  while (offset < source.length) {
    JSON_TOKEN_RE.lastIndex = offset;
    const match = JSON_TOKEN_RE.exec(source);
    if (match) {
      const token = match[0];
      const end = JSON_TOKEN_RE.lastIndex;
      parent.append(node("span", jsonTokenClass(token, source, end), token));
      offset = end;
      continue;
    }

    // Keep non-token runs as text nodes. In particular, this preserves all
    // whitespace and malformed/raw text without ever assigning innerHTML.
    const nextCandidate = source.slice(offset).search(/["{}\[\],:\-0-9tfn]/);
    const end = nextCandidate < 0 ? source.length : offset + Math.max(1, nextCandidate);
    parent.append(document.createTextNode(source.slice(offset, end)));
    offset = end;
  }
}

function createJsonPre(value) {
  const pre = node("pre", "json-highlight");
  const source = typeof value === "string" ? value : jsonText(value);
  appendJsonTokens(pre, source ?? "");
  return pre;
}

function eventText(record) {
  return `${eventType(record)} ${jsonText(record, false)}`.toLowerCase();
}

function setStatus(message, isError = false) {
  const target = $("#global-status");
  target.textContent = message || "";
  target.style.color = isError ? "var(--red)" : "var(--orange)";
}

function statusClass(status) {
  return String(status ?? "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "-");
}

function displayStatus(status) {
  return status ? String(status).replaceAll("_", " ") : "unknown";
}

function createDetails(title, value, className = "raw-json") {
  const details = node("details", className);
  const summary = node("summary", null, title);
  const pre = createJsonPre(value);
  const copy = node("button", "copy-button", "Copy");
  copy.type = "button";
  copy.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    try {
      await navigator.clipboard.writeText(pre.textContent || "");
      copy.textContent = "Copied";
      setTimeout(() => { copy.textContent = "Copy"; }, 1100);
    } catch { copy.textContent = "Copy unavailable"; }
  });
  summary.append(copy);
  details.append(summary, pre);
  return details;
}

function createTextBlock(label, text, open = false) {
  if (!text) return null;
  const details = node("details", "text-block");
  details.open = open;
  details.append(node("summary", null, label), node("pre", null, text));
  return details;
}

function chip(label, value) {
  const wrapper = node("div", "detail-chip");
  append(wrapper, node("span", null, label), node("strong", null, value ?? "—"));
  return wrapper;
}

function summaryText(record) {
  const type = eventType(record);
  if (type === "run_started") return `Scheduler v${record.config?.scheduler_version ?? "?"} · ${record.config?.model?.name ?? "model unspecified"}`;
  if (type === "scheduler_loaded") return `${record.scheduler_source ?? "scheduler"} · version ${record.scheduler_version ?? "?"}`;
  if (type === "snapshot_loaded") return `${record.snapshot_id ?? "snapshot"} · schema ${record.schema_version ?? "?"}`;
  if (type === "dataset_loaded") return `${formatNumber(record.selected_trace_count ?? record.trace_count, 0)} selected traces · split ${record.split ?? "—"}`;
  if (type === "evaluation_started") return `${formatNumber(record.selected_trace_count ?? record.trace_count, 0)} traces queued for evaluation`;
  if (type === "evaluation_completed") return `${record.status ?? "completed"} · ${formatNumber(record.trace_count, 0)} traces`;
  if (type === "diagnosis_started") return `Diagnosis for scheduler v${record.scheduler_version ?? "?"}`;
  if (["session_start", "session_end"].includes(type)) return `${record.session_id ?? "session"} · ${displayStatus(record.status ?? "started")}`;
  if (["diagnosis_agent_started", "agent_started", "model_started", "agent_context_prepared"].includes(type)) return `${record.phase ?? "diagnosis"}${record.session_id ? ` · ${record.session_id}` : ""}`;
  if (type === "round_start") return `Round ${record.round_number ?? record.round ?? "?"} begins`;
  if (type === "round_end") return `Round ${record.round_number ?? record.round ?? "?"} · ${displayStatus(record.status ?? "ended")}`;
  if (type === "llm_request") {
    const projection = projectResponseRequest(record);
    return `${projection.inputCount} input item${projection.inputCount === 1 ? "" : "s"}${projection.model ? ` · ${projection.model}` : ""}`;
  }
  if (type === "tool_call") return `${record.name ?? record.tool ?? "tool"} requested`;
  if (type === "tool_result") return `${record.name ?? record.tool ?? "tool"} · ${displayStatus(record.status ?? "returned")}`;
  if (type === "tool_evaluation") return `${record.name ?? record.tool ?? "tool"} · ${record.execution_outcome ?? record.status ?? "evaluated"}`;
  if (type === "meta_tool_started") return `${record.tool ?? "meta tool"} started`;
  if (type === "meta_tool_completed") return `${record.tool ?? "meta tool"} completed`;
  if (type === "evaluation_trace") return `DAG ${record.trace_id ?? "trace"} · ${displayStatus(record.status ?? "unscored")}`;
  if (type.endsWith("_failed") || type === "run_failed") return `${record.error_type ?? "failure"}${record.error ? ` · ${record.error}` : ""}`;
  return compactJson(Object.fromEntries(Object.entries(record).filter(([key]) => !["sequence", "timestamp", "event_type", "type"].includes(key))), 260) || "No summary fields";
}

function metricsForEvaluation(record) {
  const metrics = record.metrics ?? record.raw_metrics ?? {};
  const grid = node("div", "detail-grid");
  [["score", metrics.composite_score], ["accuracy", metrics.accuracy], ["latency", metrics.latency === undefined ? undefined : `${formatNumber(metrics.latency)} ms`], ["resource", metrics.resource === undefined ? undefined : `${formatNumber(metrics.resource)} MiB`]].forEach(([label, value]) => grid.append(chip(label, value === undefined ? "—" : value)));
  return grid;
}

function evaluationDetails(record) {
  const fragment = document.createDocumentFragment();
  const nodes = Array.isArray(record.dag?.nodes) ? record.dag.nodes : [];
  if (nodes.length) {
    const chain = nodes.map((item) => item.tool_id ?? item.node_id ?? "node").join(" → ");
    fragment.append(node("p", "summary", `DAG tool chain: ${chain}`));
  }
  fragment.append(metricsForEvaluation(record));
  if (record.reason) fragment.append(node("p", "summary", `Reason: ${record.reason}`));
  return fragment;
}

function modelDetails(record) {
  const projection = projectModelOutput(record);
  const fragment = document.createDocumentFragment();
  if (projection.toolCalls.length) {
    const calls = node("div", "model-tools");
    projection.toolCalls.forEach((call) => {
      const intent = node("section", "tool-intent");
      const head = node("div", "tool-intent-head");
      append(head, node("strong", null, call.name), node("span", "call-id", call.callId));
      intent.append(head);
      if (call.argumentsError) {
        intent.append(node("p", "summary", `Arguments error: ${call.argumentsError}`));
        intent.append(node("pre", null, `Raw arguments: ${asText(call.rawArguments)}`));
      } else {
        intent.append(node("pre", null, jsonText(call.arguments)));
      }
      calls.append(intent);
    });
    fragment.append(calls);
  }
  fragment.append(createTextBlock("Message", projection.modelText, true));
  fragment.append(createTextBlock("Reasoning", projection.reasoningText));
  if (projection.metadata.usage) {
    const usage = node("div", "detail-grid");
    const values = projection.metadata.usage;
    [["input tokens", values.input], ["output tokens", values.output], ["reasoning", values.reasoning], ["total", values.total]].forEach(([label, value]) => usage.append(chip(label, formatNumber(value, 0))));
    fragment.append(usage);
  }
  return fragment;
}

function requestDetails(record) {
  const fragment = document.createDocumentFragment();
  const projection = projectResponseRequest(record);
  const grid = node("div", "detail-grid");
  grid.append(chip("input items", projection.inputCount));
  if (projection.model) grid.append(chip("model", projection.model));
  if (projection.toolNames.length) grid.append(chip("declared tools", projection.toolNames.join(", ")));
  fragment.append(grid);
  if (record.system_prompt !== undefined) fragment.append(createTextBlock("System prompt", String(record.system_prompt)));
  if (record.input !== undefined) fragment.append(createDetails("Input payload", record.input));
  return fragment;
}

function configDetails(record) {
  const config = record.config;
  if (!config || typeof config !== "object") return null;
  const fragment = document.createDocumentFragment();
  const grid = node("div", "detail-grid");
  [["scheduler", config.scheduler_source], ["dataset", config.dataset], ["split", config.split], ["model", config.model?.name]].forEach(([label, value]) => {
    if (value !== undefined) grid.append(chip(label, value));
  });
  fragment.append(grid);
  if (config.scoring !== undefined) fragment.append(createDetails("Scoring configuration", config.scoring));
  return fragment;
}

function contextDetails(record) {
  const fragment = document.createDocumentFragment();
  ["context", "history", "available_tools", "input"].forEach((key) => {
    if (record[key] !== undefined) fragment.append(createDetails(key, record[key]));
  });
  return fragment.childNodes.length ? fragment : null;
}

function toolDetails(record) {
  const fragment = document.createDocumentFragment();
  const type = eventType(record);
  const callId = record.call_id ? String(record.call_id) : "";
  const link = callId ? state.ledger.get(callId) : null;
  const grid = node("div", "detail-grid");
  if (record.name ?? record.tool) grid.append(chip("tool", record.name ?? record.tool));
  if (callId) grid.append(chip("call_id", callId));
  if (record.tool_kind) grid.append(chip("kind", record.tool_kind));
  if (record.status) grid.append(chip("status", record.status));
  if (record.execution_outcome) grid.append(chip("outcome", record.execution_outcome));
  fragment.append(grid);
  if (type === "tool_call" && record.arguments !== undefined) fragment.append(createDetails("Arguments", record.arguments));
  if (type === "tool_result" && record.result !== undefined) {
    fragment.append(node("p", "summary", `Result: ${compactJson(record.result, 260)}`));
    fragment.append(createDetails("Result", record.result));
  }
  if (link && type !== "tool_call") fragment.append(node("p", "call-link", `Linked request: ${link.name ?? "tool"} · ${link.callId}`));
  return fragment;
}

function eventExtra(record) {
  const type = eventType(record);
  if (["llm_response", "model_output"].includes(type)) return modelDetails(record);
  if (type === "llm_request" || type === "response_request") return requestDetails(record);
  if (type === "evaluation_trace") return evaluationDetails(record);
  if (type === "run_started") return configDetails(record);
  if (["agent_context_prepared", "context"].includes(type)) return contextDetails(record);
  if (["tool_call", "tool_result", "tool_evaluation"].includes(type)) return toolDetails(record);
  if (["meta_tool_started", "meta_tool_completed"].includes(type)) {
    const fragment = document.createDocumentFragment();
    const grid = node("div", "detail-grid");
    grid.append(chip("tool", record.tool));
    if (record.result !== undefined) grid.append(chip("result", compactJson(record.result, 140)));
    fragment.append(grid);
    if (record.arguments !== undefined) fragment.append(createDetails("Arguments", record.arguments));
    if (record.result !== undefined) fragment.append(createDetails("Result", record.result));
    return fragment;
  }
  return null;
}

function eventCard(record, previous, context = {}) {
  const type = eventType(record);
  const cardClass = type.includes("failed") || type === "run_failed" ? "event-failure" : ["llm_response", "model_output"].includes(type) ? "event-model" : ["tool_call", "tool_result", "tool_evaluation"].includes(type) ? "event-tool" : "";
  const card = node("article", `event-card ${cardClass}`);
  const head = node("div", "event-head");
  const sequence = node("span", "event-sequence", `#${record.sequence ?? record.__line ?? "?"}`);
  const title = node("div", "event-title");
  let titleText = type === "llm_response" || type === "model_output" ? "Model output" : type.replaceAll("_", " ");
  if (type === "llm_response" || type === "model_output") titleText = `Model output · Requests ${projectModelOutput(record).toolCalls.length} tools`;
  append(title, node("strong", null, titleText), node("div", "event-time", `${formatTime(record.timestamp)} · ${record.timestamp ?? "timestamp unavailable"}`));
  const tags = node("div", "event-tags");
  append(tags, node("span", "event-tag", `type: ${type}`));
  const round = context.round ?? record.round_number ?? record.round;
  if (round !== undefined && round !== null) tags.append(node("span", "event-tag phase", `round ${round}`));
  if (context.phase ?? record.phase) tags.append(node("span", "event-tag phase", String(context.phase ?? record.phase)));
  if (previous && record.timestamp && previous.timestamp) {
    const gap = new Date(record.timestamp).valueOf() - new Date(previous.timestamp).valueOf();
    if (Number.isFinite(gap)) tags.append(node("span", "event-tag gap", `+${formatDuration(Math.max(0, gap))}`));
  }
  append(head, sequence, title, tags);
  const body = node("div", "event-body");
  body.append(node("p", "summary", summaryText(record)));
  const extra = eventExtra(record);
  if (extra) body.append(extra);
  body.append(createDetails("Raw JSON", record));
  card.append(head, body);
  return card;
}

function renderHeader(filename, records) {
  const metrics = deriveMetrics(records);
  const header = $("#run-header");
  header.replaceChildren();
  const title = node("div", "run-title");
  const status = node("span", `status-badge ${statusClass(metrics.status)}`);
  append(status, node("span", "status-dot", ""), node("span", null, displayStatus(metrics.status)));
  append(title, node("div", "kicker", "Session / run"), node("h2", null, filename), status, node("p", "run-subtitle", `${metrics.startTimestamp ? formatTime(new Date(metrics.startTimestamp).toISOString()) : "No start timestamp"}${metrics.durationMs !== null ? ` · ${formatDuration(metrics.durationMs)}` : ""}`));
  const stats = node("div", "header-stats");
  [["events", metrics.events], ["rounds", metrics.rounds], ["model outputs", metrics.modelOutputs], ["tool calls", metrics.toolCalls]].forEach(([label, value]) => stats.append(chip(label, formatNumber(value, 0))));
  header.append(title, stats);
}

function populateFilter(records) {
  const filter = $("#event-filter");
  const selected = filter.value;
  filter.replaceChildren(node("option", null, "All event types"));
  filter.firstChild.value = "all";
  [...new Set(records.map(eventType))].sort().forEach((type) => {
    const option = node("option", null, type);
    option.value = type;
    filter.append(option);
  });
  filter.value = [...filter.options].some((option) => option.value === selected) ? selected : "all";
}

function renderEvents() {
  if (!state.parsed) return;
  const query = $("#event-search").value.trim().toLowerCase();
  const filter = $("#event-filter").value;
  const matches = (record) => (!query || eventText(record).includes(query)) && (filter === "all" || eventType(record) === filter);
  const records = state.parsed.records.filter(matches);
  const target = $("#event-stream");
  target.replaceChildren();
  $("#event-count-label").textContent = `${records.length} shown · ${state.parsed.records.length} total`;
  if (!records.length) { target.append(node("div", "event-empty", "No events match the current search or filter.")); return; }
  groupRecordsByRound(state.parsed.records, matches).forEach((group) => {
    const section = node("section", "round-group");
    const heading = node("header", "round-group-heading");
    const title = group.round === null ? "Session events" : `Round ${group.round}`;
    heading.append(node("h3", null, title));
    if (group.phase) heading.append(node("span", "round-phase", String(group.phase)));
    heading.append(node("span", "round-count", `${group.events.length} event${group.events.length === 1 ? "" : "s"}`));
    const events = node("div", "round-events");
    group.events.forEach(({ record, context, index }) => {
      const previous = index > 0 ? state.parsed.records[index - 1] : null;
      events.append(eventCard(record, previous, context));
    });
    section.append(heading, events);
    target.append(section);
  });
}

function renderNotices() {
  const target = $("#parse-notices");
  const notices = [...(state.parsed?.errors ?? []).map((item) => `Error: ${item.message}`), ...(state.parsed?.warnings ?? [])];
  target.replaceChildren();
  target.hidden = !notices.length;
  if (notices.length) target.append(node("span", null, notices.join("\n")));
}

function renderTraceList() {
  const target = $("#trace-list");
  const query = $("#trace-search").value.trim().toLowerCase();
  target.replaceChildren();
  const traces = state.traces.filter((trace) => trace.filename.toLowerCase().includes(query));
  $("#trace-list-status").textContent = `${traces.length} of ${state.traces.length} JSONL file${state.traces.length === 1 ? "" : "s"}`;
  if (!traces.length) { target.append(node("div", "event-empty", state.traces.length ? "No matching files." : "No .jsonl traces in this directory.")); return; }
  traces.forEach((trace) => {
    const button = node("button", `trace-item ${trace.filename === state.activeFilename ? "active" : ""}`);
    button.type = "button";
    button.setAttribute("aria-label", `Open ${trace.filename}`);
    button.append(node("span", "trace-name", trace.filename));
    const meta = node("span", "trace-meta");
    const dot = node("span", `status-dot ${statusClass(trace.status)}`);
    append(meta, dot, node("span", null, displayStatus(trace.status)), node("span", null, trace.start_time ? formatTime(trace.start_time) : "no timestamp"));
    button.append(meta, node("span", "trace-counts", `${trace.event_count} events · ${trace.round_count} rounds · ${trace.tool_call_count} tools`));
    button.addEventListener("click", () => selectTrace(trace.filename));
    target.append(button);
  });
}

async function refreshTraces(selectFirst = true) {
  try {
    const response = await fetch(`/api/traces?refresh=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Trace index returned HTTP ${response.status}`);
    const payload = await response.json();
    state.traces = Array.isArray(payload) ? payload : Array.isArray(payload.traces) ? payload.traces : [];
    renderTraceList();
    if (selectFirst && state.traces.length && !state.traces.some((trace) => trace.filename === state.activeFilename)) await selectTrace(state.traces[0].filename);
    if (!state.traces.length) { $("#empty-view").hidden = false; $("#trace-view").hidden = true; }
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Unable to load traces", true);
    state.traces = [];
    renderTraceList();
  }
}

async function selectTrace(filename) {
  try {
    const response = await fetch(`/api/traces/${encodeURIComponent(filename)}?refresh=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Trace returned HTTP ${response.status}`);
    state.traceText = await response.text();
    state.parsed = parseJsonl(state.traceText);
    state.ledger = buildToolLedger(state.parsed.records);
    state.activeFilename = filename;
    $("#empty-view").hidden = true;
    $("#trace-view").hidden = false;
    renderTraceList();
    renderHeader(filename, state.parsed.records);
    populateFilter(state.parsed.records);
    renderNotices();
    renderEvents();
    setStatus("");
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Unable to load trace", true);
  }
}

$("#trace-search").addEventListener("input", renderTraceList);
$("#event-search").addEventListener("input", renderEvents);
$("#event-filter").addEventListener("change", renderEvents);
$("#refresh-traces").addEventListener("click", () => refreshTraces(false));
refreshTraces();
