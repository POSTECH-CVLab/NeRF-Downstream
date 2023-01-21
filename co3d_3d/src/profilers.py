import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
from pytorch_lightning.profiler import SimpleProfiler


class SumProfiler(SimpleProfiler):
    """
    This profiler simply records the duration of actions (in seconds) and reports
    the mean duration of each action and the total time spent over the entire training run.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        output_filename: Optional[str] = None,
    ):
        SimpleProfiler.__init__(self, dirpath=dirpath, filename=filename)
        self.reset()

    def reset(self) -> None:
        self.recorded_durations = defaultdict(float)
        self.call_counts = defaultdict(int)

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name] += duration
        self.call_counts[action_name] += 1

    def make_report(self):
        total_duration = time.monotonic() - self.start_time
        report = [
            [a, d, self.call_counts[a], 100.0 * d / total_duration]
            for a, d in self.recorded_durations.items()
        ]
        report.sort(key=lambda x: x[3], reverse=True)
        return report, total_duration

    def summary(self) -> str:
        output_string = "\n\nProfiler Report\n"

        if len(self.recorded_durations) > 0:
            max_key = np.max([len(k) for k in self.recorded_durations.keys()])

            def log_row(action, mean, num_calls, total, per):
                row = f"{os.linesep}{action:<{max_key}s}\t|  {mean:<15}\t|"
                row += f"{num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                return row

            output_string += log_row(
                "Action",
                "Mean duration (s)",
                "Num calls",
                "Total time (s)",
                "Percentage %",
            )
            output_string_len = len(output_string)
            output_string += f"{os.linesep}{'-' * output_string_len}"
            report, total_duration = self.make_report()
            output_string += log_row("Total", "-", "_", f"{total_duration:.5}", "100 %")
            output_string += f"{os.linesep}{'-' * output_string_len}"
            for action, durations, call_count, duration_per in report:
                output_string += log_row(
                    action,
                    f"{(durations / call_count):.5}",
                    f"{call_count:}",
                    f"{durations:.5}",
                    f"{duration_per:.5}",
                )

        output_string += os.linesep
        return output_string
