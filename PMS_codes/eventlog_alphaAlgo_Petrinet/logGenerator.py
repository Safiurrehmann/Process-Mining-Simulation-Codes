import csv
import random
from datetime import datetime, timedelta

def generate_event_log(process_description, num_traces, noise_level, uncommon_path_freq, missing_event_prob):
    steps = process_description.split("->")
    event_log = []
    users = ["Sara Jones", "Pete Scott", "Sue Fox", "Carol Hope"]
    products = ["iPhone5S", "iPhone4S"]
    start_time = datetime(2014, 1, 22, 9, 0)

    for trace_num in range(1, num_traces + 1):
        trace = []
        uncommon_path = random.random() < uncommon_path_freq
        current_time = start_time

        for step in steps:
            if random.random() > missing_event_prob:
                trace.append({
                    "order number": trace_num,
                    "activity": step if not (uncommon_path and random.random() < 0.3) else step,
                    "timestamp": current_time.strftime("%d-%m-%Y@%H.%M"),
                    "user": random.choice(users),
                    "product": random.choice(products),
                    "quantity": random.randint(1, 3)
                })
                current_time += timedelta(minutes=random.randint(5, 20))

        # Add noise events
        for _ in range(int(noise_level * len(steps))):
            noise_event = {
                "order number": trace_num,
                "activity": "Noise_" + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                "timestamp": current_time.strftime("%d-%m-%Y@%H.%M"),
                "user": random.choice(users),
                "product": random.choice(products),
                "quantity": random.randint(1, 3)
            }
            trace.insert(random.randint(0, len(trace)), noise_event)

        event_log.extend(trace)

    # Write to CSV
    with open("event_log.csv", "w", newline="") as csvfile:
        fieldnames = ["order number", "activity", "timestamp", "user", "product", "quantity"]
        log_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        log_writer.writeheader()
        for entry in event_log:
            log_writer.writerow(entry)

    print("Event log generated and saved as 'event_log.csv'")

# Example function call
generate_event_log(
    process_description="register order->check stock->ship order->handle payment",
    num_traces=1000,
    noise_level=0.1,
    uncommon_path_freq=0.3,
    missing_event_prob=0.1
)
