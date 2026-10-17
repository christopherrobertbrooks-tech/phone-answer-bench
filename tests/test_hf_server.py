"""hf_server's plain parts (no model): reading a search request out of a reply, and the messages the templates get."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import answers  # noqa: E402
import hf_server as S  # noqa: E402


class FindCall(unittest.TestCase):
    def test_forms_that_are_a_search(self):
        self.assertEqual(S.find_call('web_search("Bears score last night")'), "Bears score last night")
        self.assertEqual(S.find_call("Let me check. web_search(query='weather in Chicago')"), "weather in Chicago")
        self.assertEqual(S.find_call('Web_search("how long to cook spaghetti")'), "how long to cook spaghetti")  # MobileMoE-M
        self.assertEqual(S.find_call('{"name": "web_search", "arguments": {"query": "gas prices"}}'), "gas prices")
        self.assertEqual(S.find_call('{"tool_call": {"name": "web_search", "arguments": {"query": "x y"}}}'), "x y")
        self.assertEqual(S.find_call('{"name": "web_search", "arguments": "{\\"query\\": \\"z\\"}"}'), "z")

    def test_forms_that_are_not(self):
        self.assertIsNone(S.find_call("Mary Shelley wrote Frankenstein."))
        self.assertIsNone(S.find_call("I can search the web for that if you like."))
        self.assertIsNone(S.find_call('{"name": "calculator", "arguments": {"x": 1}}'))


class Reply(unittest.TestCase):
    def test_a_call_goes_back_as_a_tool_call(self):
        m = S.reply('web_search("Bears score")', 100, 8, 0.5)["choices"][0]["message"]
        self.assertEqual((m["content"], m["tool_calls"][0]["function"]["name"], json.loads(m["tool_calls"][0]["function"]["arguments"])),
                         ("", "web_search", {"query": "Bears score"}))

    def test_words_go_back_as_words(self):
        self.assertEqual(S.reply("  It's Paris.  ", 50, 4, 0.2)["choices"][0]["message"], {"role": "assistant", "content": "It's Paris."})

    def test_timings_like_llama_servers(self):
        self.assertEqual(S.reply("x", 50, 10, 0.5)["timings"], {"prompt_n": 50, "predicted_n": 10, "predicted_per_second": 20.0})


class Plain(unittest.TestCase):
    def test_a_result_item_as_these_templates_get_it(self):
        item = {"kind": "result", "say": "Will it snow?", "query": "snow forecast", "result": "No snow expected."}
        p = S.plain(answers.messages(item))
        self.assertEqual([x["role"] for x in p], ["system", "user", "assistant", "user"])
        self.assertTrue(p[0]["content"].startswith(answers.SYSTEM) and p[0]["content"].count("web_search(") == 1)
        self.assertEqual(p[2]["content"], 'web_search("snow forecast")')
        self.assertEqual(p[3]["content"], "Search results:\nNo snow expected.")


if __name__ == "__main__":
    unittest.main()
