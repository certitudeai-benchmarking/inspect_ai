import { Capabilities } from "../types";
import { asyncJsonParse } from "../utils/json-worker";
import { download_file } from "./api-shared";
import { LogContents, LogViewAPI } from "./types";

const loaded_time = Date.now();
let last_eval_time = 0;

async function client_events() {
  const params = new URLSearchParams();
  params.append("loaded_time", String(loaded_time.valueOf()));
  params.append("last_eval_time", String(last_eval_time.valueOf()));
  return (await api("GET", `/api/events?${params.toString()}`)).parsed;
}

async function eval_logs() {
  const logs = await api("GET", `/api/logs`);
  last_eval_time = Date.now();
  return logs.parsed;
}

async function eval_log(
  file: string,
  headerOnly?: number,
  _capabilities?: Capabilities,
): Promise<LogContents> {
  return await api(
    "GET",
    `/api/logs/${encodeURIComponent(file)}?header-only=${headerOnly}`,
  );
}

async function eval_log_size(file: string): Promise<number> {
  return (await api("GET", `/api/log-size/${encodeURIComponent(file)}`)).parsed;
}

async function eval_log_bytes(file: string, start: number, end: number) {
  return await api_bytes(
    "GET",
    `/api/log-bytes/${encodeURIComponent(file)}?start=${start}&end=${end}`,
  );
}

async function eval_log_headers(files: string[]) {
  const params = new URLSearchParams();
  for (const file of files) {
    params.append("file", file);
  }
  return (await api("GET", `/api/log-headers?${params.toString()}`)).parsed;
}

async function api(
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
  body?: string,
) {
  // build headers
  const headers: HeadersInit = {
    Accept: "application/json",
    Pragma: "no-cache",
    Expires: "0",
    ["Cache-Control"]: "no-cache",
  };
  if (body) {
    headers["Content-Type"] = "application/json";
  }

  // make request
  const response = await fetch(`${path}`, { method, headers, body });
  if (response.ok) {
    const text = await response.text();
    return {
      parsed: await asyncJsonParse(text),
      raw: text,
    };
  } else if (response.status !== 200) {
    const message = (await response.text()) || response.statusText;
    const error = new Error(`Error: ${response.status}: ${message})`);
    throw error;
  } else {
    throw new Error(`${response.status} - ${response.statusText} `);
  }
}

async function api_bytes(
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
) {
  // build headers
  const headers: HeadersInit = {
    Accept: "application/octet-stream",
    Pragma: "no-cache",
    Expires: "0",
    ["Cache-Control"]: "no-cache",
  };

  // make request
  const response = await fetch(`${path}`, { method, headers });
  if (response.ok) {
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
  } else if (response.status !== 200) {
    const message = (await response.text()) || response.statusText;
    const error = new Error(`Error: ${response.status}: ${message})`);
    throw error;
  } else {
    throw new Error(`${response.status} - ${response.statusText} `);
  }
}

async function open_log_file() {
  // No op
}

const browserApi: LogViewAPI = {
  client_events,
  eval_logs,
  eval_log,
  eval_log_size,
  eval_log_bytes,
  eval_log_headers,
  download_file,
  open_log_file,
};
export default browserApi;
