import argparse
import asyncio
import statistics
import time

import httpx


async def run(base_url: str, count: int, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)
    durations = []
    successes = 0

    async with httpx.AsyncClient(timeout=30) as client:
        async def execute(index):
            nonlocal successes
            async with semaphore:
                start = time.perf_counter()
                response = await client.post(
                    f"{base_url}/v1/infer",
                    json={"prompt": f"Benchmark request {index}", "max_new_tokens": 16},
                )
                if response.status_code != 202:
                    return
                job_id = response.json()["job_id"]
                for _ in range(100):
                    result = await client.get(f"{base_url}/v1/jobs/{job_id}")
                    if result.status_code == 200 and result.json()["status"] in ("completed", "failed"):
                        durations.append(time.perf_counter() - start)
                        successes += result.json()["status"] == "completed"
                        return
                    await asyncio.sleep(0.1)

        started = time.perf_counter()
        await asyncio.gather(*(execute(i) for i in range(count)))
        elapsed = time.perf_counter() - started
    print(f"Submitted: {count}; completed: {successes}; elapsed: {elapsed:.3f}s")
    if durations:
        ordered = sorted(durations)
        print(f"Throughput: {successes / elapsed:.2f} completed jobs/s")
        print(f"Mean latency: {statistics.mean(durations):.3f}s")
        print(f"P95 latency: {ordered[max(0, int(0.95 * len(ordered)) - 1)]:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.requests, args.concurrency))
