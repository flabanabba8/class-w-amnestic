Read the integer target from the task input (e.g. "count to 5").
Keep the current count in `environment.counter`.
Each step: if counter < target, set counter = counter + 1 and use the `continue` action.
When counter == target, submit a completion with outcome=success and the final count in `final_answer`.
Never add facts without evidence; the count is bookkeeping and lives in `environment`.
