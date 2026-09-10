import json
import sys
from datetime import datetime
from time import sleep

import requests


sys.stdout.reconfigure(encoding="utf-8")


BASE_URL = "http://127.0.0.1:2026"
ROBOT_ID = "202610038037"
ARENA_ID = "default"
TIMEOUT_SECONDS = 5
MAX_RETRIES = 3
LOG_FILE = "log.txt"
RUN_ID = datetime.now().strftime("%Y%m%d%H%M%S")


def now_text():
    return datetime.now().isoformat(timespec="seconds")


def write_log(message):
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{now_text()}] {message}\n")


def post_with_retry(path, payload):
    url = BASE_URL + path

    for attempt in range(1, MAX_RETRIES + 1):
        write_log(f"REQUEST attempt={attempt} path={path} payload={json.dumps(payload, ensure_ascii=False)}")

        try:
            response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
            response_json = response.json()
            write_log(f"RESPONSE attempt={attempt} status={response.status_code} json={json.dumps(response_json, ensure_ascii=False)}")
            return response_json
        except requests.RequestException as error:
            write_log(f"ERROR attempt={attempt} path={path} error={repr(error)}")
            if attempt == MAX_RETRIES:
                raise
            sleep(1)
        except ValueError as error:
            write_log(f"ERROR attempt={attempt} path={path} invalid_json={repr(error)} text={response.text}")
            if attempt == MAX_RETRIES:
                raise
            sleep(1)


def base_payload(request_id):
    return {
        "arena_id": ARENA_ID,
        "robot_id": ROBOT_ID,
        "request_id": request_id,
    }


def main():
    try:
        enter_response = post_with_retry("/enter", base_payload(f"enter-{RUN_ID}"))
        print(json.dumps(enter_response, ensure_ascii=False, indent=2))

        post_with_retry("/exit", base_payload(f"exit-{RUN_ID}"))
    except requests.RequestException as error:
        print(f"请求失败，已重试 {MAX_RETRIES} 次：{error}")
    except ValueError as error:
        print(f"响应不是合法 JSON，已重试 {MAX_RETRIES} 次：{error}")


if __name__ == "__main__":
    main()
