"""DQ Report."""

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
# from user_input.metrics import Metric
from checklist import CHECKLIST
from metrics import Metric

import pandas as pd

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report."""
        self.report_ = {}
        report = self.report_

        # Check if engine supported
        if self.engine != "pandas":
            raise NotImplementedError("Only pandas API currently supported!")
        
        # Create base dataframe from checklist
        report = pd.DataFrame().from_dict(
            data=self.checklist
        )
        report.columns = ['table_name', 'metric', 'limits']

        # Append metric, status and error values for each metric in chechlist
        metrics= []
        errors = []
        status = []
        for row in report.values:
            if row[0] == 'relevance':
                date = tables['relevance']
                try:
                    metric = row[1](df=date)
                    metrics_series.append(metric)

                    if metric not in row[2].values():
                        status_series.append('F')
                    else:
                        status_series.append('.')
                except Exception as e:
                    error_series.append(e)
                    status_series.append('E')
            elif row[0] == 'sales':
                date = tables['sales']
                try:
                    metric = row[1](df=date)
                    metrics_series.append(metric)

                    if metric not in row[2].values():
                        status_series.append('F')
                    else:
                        status_series.append('.')
                except Exception as e:
                    error_series.append(e)
                    status_series.append('E')
        report['values'] = pd.Series(
            data=metrics_series
        )
        report['status'] = pd.Series(
            data=status_series
        )
        report['error'] = pd.Series(
            data=error_series
        )

        print(report.columns)
        self.report_ = report.to_dict()
        report = report.to_dict()

        return report

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )


if __name__ == '__main__':
    sales = pd.read_csv('JUNIOR/DQ_AND_PYSPARK/ke_daily_sales.csv')
    relevance = pd.read_csv('JUNIOR/DQ_AND_PYSPARK/ke_visits.csv')

    report = Report(
        checklist=CHECKLIST
    )
    print(report.fit({
        'sales': sales,
        'relevance': relevance
    }))
    print(report.to_str())


    # print(CHECKLIST[8][1](df=relevance))