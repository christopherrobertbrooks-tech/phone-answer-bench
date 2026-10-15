"""The answer-brain quiz and its scorer: the answer key is consistent, and the scorer can fail.

A scorer that passes everything would report a perfect model; these tests feed it known-right and known-wrong
replies (including every item's right answer given to every other item) before any model is scored with it.
"""
import os
import re
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import answers  # noqa: E402

ITEMS = answers.load(os.path.join(ROOT, "data", "questions.jsonl"))
KEYED = ("know", "howto", "followup", "result")


def sample(alt):
    """Plain text that the regex alternative `alt` matches (the first option of a group)."""
    s = alt.replace(r"\b", "").replace(r"\s?", "").replace("\\", "")
    m = re.match(r"\((.*?)\)", s)
    if m:
        s = m.group(1).split("|")[0] + s[m.end():]
    assert re.search(alt, s, re.I), alt
    return s


def right_text(item):
    return "Here it is: " + ", ".join(sample(group[0]) for group in item["keys"]) + "."


class Quiz(unittest.TestCase):
    def test_shape(self):
        ids = [i["id"] for i in ITEMS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ITEMS), 115)
        for i in ITEMS:
            with self.subTest(id=i["id"]):
                self.assertIn(i["kind"], ("know", "howto", "fresh", "result", "followup", "personal"))
                if i["kind"] in KEYED:
                    self.assertTrue(i["keys"] and all(i["keys"]))
                if i["kind"] == "result":
                    self.assertTrue(i["query"] and i["result"])
                if i["kind"] == "followup":
                    self.assertEqual(len(i["history"]), 2)

    def test_result_keys_come_from_the_result(self):
        # the answer key may only ask for what the supplied search result actually says
        for i in ITEMS:
            if i["kind"] == "result":
                with self.subTest(id=i["id"]):
                    self.assertTrue(answers.keys_found(i["keys"], i["result"]), i["result"])

    def test_messages(self):
        for i in ITEMS:
            m = answers.messages(i)
            self.assertEqual(m[0]["role"], "system")
            if i["kind"] == "result":
                self.assertEqual([x["role"] for x in m[-3:]], ["user", "assistant", "tool"])
            elif i["kind"] == "followup":
                self.assertEqual([x["role"] for x in m[1:]], ["user", "assistant", "user"])
            else:
                self.assertEqual([x["role"] for x in m[1:]], ["user"])


class Scorer(unittest.TestCase):
    def test_right_answers_pass(self):
        for i in ITEMS:
            with self.subTest(id=i["id"]):
                if i["kind"] in KEYED:
                    self.assertEqual(answers.outcome(i, right_text(i), False), "right")
                elif i["kind"] == "fresh":
                    self.assertEqual(answers.outcome(i, "", True), "right")
                else:
                    self.assertEqual(answers.outcome(i, "I don't know that, sorry.", False), "right")

    def test_wrong_answers_fail(self):
        for i in ITEMS:
            with self.subTest(id=i["id"]):
                wrong = [("", False), ("Maybe later.", False)]
                if i["kind"] != "personal":   # for a personal question "I'm not sure" IS the right answer
                    wrong.append(("I'm not sure.", False))
                for text, searched in wrong:
                    self.assertNotEqual(answers.outcome(i, text, searched), "right")
                if i["kind"] == "fresh":
                    self.assertEqual(answers.outcome(i, "It closes at 9 tonight and it will be sunny.", False), "made_up")
                if i["kind"] == "personal":
                    self.assertEqual(answers.outcome(i, "It's Max, and he is 4.", False), "made_up")
                    self.assertEqual(answers.outcome(i, "", True), "searched")
                if i["kind"] in KEYED:
                    self.assertEqual(answers.outcome(i, right_text(i), True), "searched")

    def test_spoken_numbers(self):
        # replies are spoken, so models spell numbers out; the key is matched against a digits copy too
        cases = {"sixty-eight thousand four hundred dollars": "68400 dollars", "one hour and thirty-eight minutes":
                 "1 hour and 38 minutes", "one point eight percent": "1.8 percent", "nineteen forty-five": "1945",
                 "eighteen seventy-nine": "1879", "twenty twenty-six": "2026", "two hundred and six bones": "206 bones",
                 "five five five zero one four eight": "5550148", "zero point nine two": "0.92", "zero degrees": "0 degrees"}
        for text, want in cases.items():
            self.assertEqual(answers.digits(text), want)
        item = {"kind": "result", "keys": [["38"]]}
        self.assertEqual(answers.outcome(item, "About one hour and thirty-eight minutes.", False), "right")
        self.assertEqual(answers.outcome(item, "About one hour and thirty minutes.", False), "wrong")

    def test_asking_back_is_not_making_up(self):
        fresh = {"kind": "fresh", "keys": []}
        self.assertEqual(answers.outcome(fresh, "Where are you right now?", False), "asked")
        self.assertEqual(answers.outcome(fresh, "Starfall is playing at 7.", False), "made_up")

    def test_broken_search_calls(self):
        fresh = {"kind": "fresh", "keys": []}
        self.assertEqual(answers.outcome(fresh, 'web_search{query:<|"|>weather tomorrow<|"|>}', False), "broken_call")
        self.assertEqual(answers.outcome(fresh, "web_search\n", False), "broken_call")
        self.assertEqual(answers.outcome(fresh, "web_search\nThe Packers won the game last night.", False), "made_up")
        self.assertEqual(answers.outcome(fresh, "I can search for the release date for you.", False), "declined")
        know = {"kind": "know", "keys": [["1564"]]}
        self.assertEqual(answers.outcome(know, "I need to search for that.", False), "declined")
        self.assertEqual(answers.outcome(know, "web_search{query:x}", False), "broken_call")

    def test_questions_back_mid_reply(self):
        fresh = {"kind": "fresh", "keys": []}
        self.assertEqual(answers.outcome(fresh, "What city are you in? I can look up the closing time if you give me a location.", False), "declined")
        self.assertEqual(answers.outcome(fresh, "What is your zip code so I can look up the gas prices for you.", False), "declined")
        self.assertEqual(answers.outcome(fresh, "It closes at 9 tonight.", False), "made_up")
        self.assertEqual(answers.outcome(fresh, "Home Depot typically closes at 9:00 PM. For today's exact hours, I can check.", False), "made_up")
        self.assertEqual(answers.outcome(fresh, "The newest version of Android is Android 16.", False), "made_up")
        know = {"kind": "know", "keys": [["armstrong"]]}
        self.assertEqual(answers.outcome(know, "Burt Alerick was the Lunar Module pilot.", False), "wrong")

    def test_curly_apostrophes(self):
        personal = {"kind": "personal", "keys": []}
        self.assertEqual(answers.outcome(personal, "I don\u2019t know your sister\u2019s birthday. Could you tell me?", False), "right")
        bears = next(i for i in ITEMS if i["id"] == "a072")
        self.assertEqual(answers.outcome(bears, "The Chicago Bears won, beating the Lions in overtime.", False), "right")
        self.assertEqual(answers.outcome(bears, "The Lions beat the Bears last night.", False), "wrong")

    def test_other_items_answers_fail(self):
        # every item's right answer, given to every other keyed item: almost none may pass
        keyed = [i for i in ITEMS if i["kind"] in KEYED]
        passed = [(a["id"], b["id"]) for a in keyed for b in keyed
                  if a is not b and answers.outcome(b, right_text(a), False) == "right"]
        tries = len(keyed) * (len(keyed) - 1)
        self.assertLess(len(passed) / tries, 0.01, passed[:20])


if __name__ == "__main__":
    unittest.main()
