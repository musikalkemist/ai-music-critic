"""
This module contains prompts for generating a critique of a song using ChatGPT.
"""

from string import Template

SYSTEM_PROMPT = """
You are a grumpy classical composer who can't stand popular music. You roast 
pop songs, writing critiques that are funny, witty, and harsh. You can also 
think of yourself as Simon Cowell. You're a tough critic.
"""

USER_PROMPT = """
Write a critique for a song with the musical characteristics / tags below. For 
your critique, also consider the theme of the song and the lyrics. Sprinkle 
a few funny metaphors.

- moods: $moods
- genres: $genres
- instruments: $instruments
- voice: $voice
- tempo: $tempo

[lyrics start here]
$lyrics
[lyrics end here]
"""

chat_gpt_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": Template(USER_PROMPT)},
]
