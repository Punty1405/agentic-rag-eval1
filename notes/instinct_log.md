"What I saw: hit_rate=0.5 means moderate performance. What was deeper: with denominator 2, it specifically means one of two sub-queries succeeded. Bridge: read the math constraints, not just the value."

"What I saw: zero-hit queries failed because they're complex. What was deeper: they specifically require inference beyond retrieval — derive a letter, chain facts to find a capital. Bridge: ask 'what cognitive work is this query asking for?', not just 'is this query hard?"

"When a population of queries scores uniformly poorly, the first hypothesis to check isn't 'system is broken' — it's 'are these queries even in scope for the system'. Out-of-scope queries pollute aggregate metrics and hide real signal."

"When a metric value clusters at a specific fraction (0.5, 0.33, 0.66), check the denominator. Discrete fractions reveal structural failure modes — e.g., 'k of N sub-tasks succeeded' rather than continuous performance variation."

"When a library has both 'evaluate()' batch API and per-metric .score() API, and one is deprecated — the non-deprecated path may use a different pattern. Don't assume newer = drop-in replacement. Read which API the new function is designed for."

"Negation queries ('does not mention', 'fails to include') are edge cases for both literal and semantic evaluation. Both evaluators score them at chance level."