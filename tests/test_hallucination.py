"""Offline checks for the key-fact correctness/hallucination classifier."""

import unittest

from hallucination import accepted_forms, classify_response, extract_key_fact, fact_rates


class HallucinationChecks(unittest.TestCase):
    def test_key_fact_extraction_for_both_templates(self) -> None:
        self.assertEqual(
            extract_key_fact("C:\\corpus\\question_cap-001.wav", "తెలంగాణ రాజధాని హైదరాబాద్."),
            "హైదరాబాద్",
        )
        self.assertEqual(
            extract_key_fact("corpus/question_sum-007.wav", "రెండు మరియు మూడు కలిపితే ఐదు వస్తుంది."),
            "ఐదు",
        )
        with self.assertRaisesRegex(ValueError, "No key-fact rule"):
            extract_key_fact("corpus/other.wav", "ఏదో సమాధానం.")
        with self.assertRaisesRegex(ValueError, "sum answer template"):
            extract_key_fact("corpus/question_sum-001.wav", "ఐదు.")

    def test_three_way_classification(self) -> None:
        self.assertEqual(classify_response("రాజధాని హైదరాబాద్ నగరం.", "హైదరాబాద్"), "correct")
        self.assertEqual(classify_response("నాకు తెలియదు.", "హైదరాబాద్"), "abstained")
        self.assertEqual(classify_response("రాజధాని చెన్నై.", "హైదరాబాద్"), "hallucinated")
        self.assertEqual(classify_response("", "హైదరాబాద్"), "hallucinated")

    def test_sum_digit_form_counts_but_only_as_a_whole_number(self) -> None:
        path = "c/question_sum-003.wav"
        forms = accepted_forms(path, "ఐదు")
        self.assertEqual(forms, ("ఐదు", "5"))
        self.assertEqual(classify_response("జవాబు 5.", forms), "correct")
        self.assertEqual(classify_response("జవాబు 15.", forms), "hallucinated")
        self.assertEqual(accepted_forms("c/question_cap-001.wav", "చెన్నై"), ("చెన్నై",))

    def test_rates_over_report_rows(self) -> None:
        rows = [
            {"audio_path": "c/question_cap-001.wav", "ground_truth_response": "తెలంగాణ రాజధాని హైదరాబాద్.",
             "asr_response": "హైదరాబాద్", "clean_transcript_response": "తెలియదు"},
            {"audio_path": "c/question_cap-002.wav", "ground_truth_response": "తమిళనాడు రాజధాని చెన్నై.",
             "asr_response": "ముంబై", "clean_transcript_response": "చెన్నై"},
        ]
        asr = fact_rates(rows, "asr_response")
        self.assertEqual((asr["correct"], asr["hallucinated"], asr["hallucination_rate"]), (1, 1, 0.5))
        clean = fact_rates(rows, "clean_transcript_response")
        self.assertEqual((clean["correct"], clean["abstained"], clean["hallucination_rate"]), (1, 1, 0.0))


if __name__ == "__main__":
    unittest.main()
