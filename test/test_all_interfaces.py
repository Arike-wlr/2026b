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


def base_payload(request_name):
    return {
        "arena_id": ARENA_ID,
        "robot_id": ROBOT_ID,
        "request_id": f"{request_name}-{RUN_ID}",
    }


def action_payload(request_name, x, y, channel):
    payload = base_payload(request_name)
    payload["position"] = {"x": x, "y": y}
    payload["channel"] = channel
    return payload


def print_response(path, response_json):
    print(path)
    print(json.dumps(response_json, ensure_ascii=False, indent=2))


def main():
    try:
        requests_to_send = [
            ("/enter", base_payload("enter")),
            ("/measure", action_payload("measure", 300, 400, 1)),
            ("/clear", action_payload("clear", 300, 0, 3)),
            ("/exit", base_payload("exit")),
        ]

        for path, payload in requests_to_send:
            response_json = post_with_retry(path, payload)
            print_response(path, response_json)

            if response_json.get("accepted") is not True:
                print(f"{path} 未被模拟器接受，停止后续测试。")
                break
    except requests.RequestException as error:
        print(f"请求失败，已重试 {MAX_RETRIES} 次：{error}")
    except ValueError as error:
        print(f"响应不是合法 JSON，已重试 {MAX_RETRIES} 次：{error}")


if __name__ == "__main__":
    main()
