import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from chunker import chunk_by_datetime


class ChunkByDatetimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "dt": pd.to_datetime(
                    [
                        "2023-01-01 00:00:01",
                        "2023-01-01 00:00:01",
                        "2023-01-01 00:00:02",
                        "2023-01-01 00:00:02",
                        "2023-01-01 00:00:02",
                        "2023-01-01 00:00:03",
                    ]
                ),
                "payload": ["a", "b", "c", "d", "e", "f"],
            },
            index=[10, 11, 12, 13, 14, 15],
        )

    def test_sizes_from_assignment(self) -> None:
        for requested, expected in [
            (1, [2, 3, 1]),
            (2, [2, 3, 1]),
            (3, [5, 1]),
            (5, [5, 1]),
            (6, [6]),
            (10, [6]),
        ]:
            with self.subTest(requested=requested):
                chunks = list(chunk_by_datetime(self.frame, "dt", requested))
                self.assertEqual([len(chunk) for chunk in chunks], expected)
                assert_frame_equal(pd.concat(chunks), self.frame)

    def test_unsorted_dates_and_other_columns(self) -> None:
        frame = self.frame.iloc[[4, 0, 5, 2, 1, 3]].copy()
        chunks = list(chunk_by_datetime(frame, "dt", 2))

        self.assertEqual([len(chunk) for chunk in chunks], [2, 3, 1])
        self.assertEqual(
            [chunk["payload"].tolist() for chunk in chunks],
            [["a", "b"], ["e", "c", "d"], ["f"]],
        )
        self.assertEqual(
            sum(len(chunk) for chunk in chunks), len(frame)
        )
        self.assertEqual(
            sum(chunk["dt"].nunique() for chunk in chunks), frame["dt"].nunique()
        )

    def test_one_large_timestamp_group_is_not_split(self) -> None:
        frame = pd.DataFrame(
            {"dt": pd.to_datetime(["2023-01-01"] * 7 + ["2023-01-02"])}
        )
        self.assertEqual(
            [len(chunk) for chunk in chunk_by_datetime(frame, "dt", 3)], [7, 1]
        )

    def test_unique_dates_hit_requested_size(self) -> None:
        frame = pd.DataFrame({"dt": pd.date_range("2023-01-01", periods=7)})
        chunks = list(chunk_by_datetime(frame, "dt", 3))
        self.assertEqual([len(chunk) for chunk in chunks], [3, 3, 1])
        assert_frame_equal(pd.concat(chunks), frame)

    def test_timezone_aware_dates(self) -> None:
        frame = self.frame.copy()
        frame["dt"] = frame["dt"].dt.tz_localize("UTC")
        frame = frame.iloc[[4, 0, 5, 2, 1, 3]]
        chunks = list(chunk_by_datetime(frame, "dt", 3))
        self.assertEqual([len(chunk) for chunk in chunks], [5, 1])

    def test_empty_input(self) -> None:
        self.assertEqual(
            list(chunk_by_datetime(self.frame.iloc[:0], "dt", 2)), []
        )

    def test_invalid_inputs(self) -> None:
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(chunk_by_datetime(self.frame, "dt", size))
        with self.assertRaises(KeyError):
            list(chunk_by_datetime(self.frame, "missing", 2))
        with self.assertRaises(TypeError):
            list(chunk_by_datetime(self.frame, "payload", 2))
        frame_with_nat = self.frame.copy()
        frame_with_nat.loc[10, "dt"] = pd.NaT
        with self.assertRaises(ValueError):
            list(chunk_by_datetime(frame_with_nat, "dt", 2))


if __name__ == "__main__":
    unittest.main()
