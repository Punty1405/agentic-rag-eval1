# RAGAS calibration script
# Runs existing pipeline on 5 Polutaion B (0.3 <= hitrate <= 0.7) queries, scores with RAGAS context_recall

import os
import sys
import json

from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import ContextRecall
from ragas.llms import llm_factory
from openai import AsyncOpenAI

# Facilitating project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.query_analyser import QueryAnalyser
from agent.query_decomposer import QueryDecomposer
from agent.retriever import Retriever
from evaluator.retrieval_evaluator import RetrievalEvaluator

# Paths to data folders
BASE_DIR = Path(__file__).parent.parent
REFERENCE_REPO = Path("~/agentic-rag-eval/reference-repo").expanduser()

CORPUS_PATH = str(REFERENCE_REPO / "dataset/corpus.json")
DATASET_PATH = str(REFERENCE_REPO / "dataset/MultiHopRAG.json")

OUTPUT_DIR = BASE_DIR / 'output'

# 5 queries from trace_analyser output - of Population Type B
TARGET_QUERIES = [
    "Who, according to articles in Sporting News, stand to make a profit by predicting outcomes such as a team's lead at the end of a quarter or the total points scored, and can also capitalize on event hype, like putting $130 on the Cowboys to potentially gain $100?",
    "Who is the individual that was once likened to a prominent investor, admitted to challenges in overseeing a rapidly expanding crypto company, faced allegations of fraud in a legal setting, and discussed corporate governance intentions with a venture firm, as reported by TechCrunch, The Verge, Cnbc, and TechCrunch respectively?",
    "Who is the individual who, after Judge Lewis Kaplan's intervention, admitted to being informed about a financial discrepancy and is also alleged by the prosecution to have knowingly committed fraud, as reported by The Verge and TechCrunch?",
    "After the report by Fortune on October 4, 2023, regarding Sam Bankman-Fried's alleged use of Caroline Ellison as a front at Alameda Research, and the subsequent report by TechCrunch involving Sam Bankman-Fried's alleged motives for committing fraud, is the portrayal of Sam Bankman-Fried's actions by both news sources consistent?",
    "Who is the individual implicated in instructing Caroline Ellison to use $14 billion of customer funds to repay debts, as reported by TechCrunch, and is also alleged by the prosecution to have committed fraud for personal gain, according to another article from TechCrunch?",
    "Has the stance of the Federal Reserve on interest rates as reported by The Sydney Morning Herald remained the same between the article published on October 1, 2023, suggesting smaller future rate cuts, and the one from November 5, 2023, indicating a continuation of rate increases?",
    "Does the Sporting News article rank Tyreek Hill as the top wide receiver for Week 14, while The Guardian article focuses on his performance in a specific game, and does the other Sporting News article question his ability to achieve less than 1,000 receiving yards for the season based on the strength of the Miami Dolphins' remaining opponents' pass defenses?",
    "Which M.L.B. superstar, who had a dozen teams interested in him during his free agency according to The New York Times, may not benefit from an agreement during the winter meetings as reported by Sporting News?",
    "Did the portrayal of Sam Bankman-Fried's financial actions remain the same in the TechCrunch report on his intentions with his wealth published on October 2, 2023, and the TechCrunch report on the allegations against him published on October 7, 2023, and is the reporting on Sam Bankman-Fried's awareness of financial issues inconsistent between the TechCrunch report on allegations against him and The Verge report on his knowledge of financial discrepancies?",
    "Does the 'Sporting News' article on the USA squad fail to mention Johnny Cardoso's inclusion after an injury similar to how 'Sporting News' reports on Lucas Cavallini's withdrawal due to injury from the Canada squad?",
    "Between the report from Fortune on Sam Bankman-Fried's use of Caroline Ellison as a front at Alameda Research published on October 4, 2023, and the TechCrunch report alleging Sam Bankman-Fried's instructions to Caroline Ellison to take customer funds published on October 6, 2023, was there consistency in the portrayal of Sam Bankman-Fried's involvement in the misuse of customer funds?",
    "Was there no change in the portrayal of Google's business practices with respect to their impact on other companies between the report by The Verge on Apple's defense of its Google Search deal published on September 26, 2023, and the report by TechCrunch on the class action antitrust suit against Google published on December 15, 2023?",
    "Between the report from The Verge on Apple's defense of its Google Search deal published on September 26, 2023, and the TechCrunch article detailing what was learned about the Google antitrust case involving Apple published on October 31, 2023, was there a consistency in the portrayal of Apple's actions regarding its choice of search engine and browser options for iPhone users?",
    "Who is the pop star that was rumored to have a secret start to her relationship with a Chiefs TE, known for experiencing major events privately and for not letting paparazzi affect her, and was also the subject of a story on CBSSports.com where a friendship bracelet played a role?",
    "Does 'The New York Times' article suggest that Emma Hayes is committed to her current role at Chelsea for the remainder of the season, in contrast to the 'Sporting News' article which discusses Graham Potter's tenure at Chelsea as being unsuccessful?",
    "Considering the excerpts from TechCrunch, has the focus of European AI startups on regulation and compliance changed since the report on November 9, 2023, before the EU lawmakers reached a deal on AI rules on December 9, 2023?",
    "Before the report from CBSSports.com on October 12, 2023, suggesting an expression of interest from Travis Kelce to Taylor Swift, and the article from The Independent - Life and Style on December 6, 2023, revealing Taylor Swift's openness about her relationship with Travis Kelce, has the narrative regarding the rumored romance between the pop star and the Chiefs TE remained consistent?",
    "Which individual is at the center of legal proceedings where he is depicted variably as a fraudulent actor by the prosecution, as per TechCrunch, and is accused of instructing a $14 billion misappropriation of customer funds, as well as using $1 billion to buy out a competitor, all while his legal representation contrasts this narrative in a trial covered by Fortune?",
    "Does 'The Guardian' article on the Sydney Swans' game day experience focus on different aspects of fan engagement compared to 'The Guardian' article discussing Mikel Arteta's comments on the importance of enduring challenging moments in a game?",
    "Between the TechCrunch report on Sam Bankman-Fried's trial published on October 1, 2023, and the TechCrunch report on the allegations against Sam Bankman-Fried published on October 7, 2023, was there consistency in the portrayal of Sam Bankman-Fried's actions related to the FTX collapse?",
    "Was the reporting on Valve's improvements to the Steam Deck hardware inconsistent after the Polygon report on Valve's updates to the Steam Deck hardware published on a date other than November 9, 2023, and the Engadget review of the Steam Deck OLED version published on the same date?",
    "Does 'The Guardian' article suggest that Manchester United is striving to emulate a team like Bayern Munich, while the 'Sporting News' article indicates that Manchester United has been eliminated from European competitions by Bayern, thus implying a current disparity in team performance?",
    "Who is the individual implicated by Fortune and multiple TechCrunch articles as having used a colleague as a front for unauthorized access to customer funds, was once likened to a prominent investor, allegedly directed the misappropriation of billions to settle debts, and is accused by prosecutors of committing fraud for personal gain?",
    "Was there a discrepancy in the reporting of Google's anticompetitive practices between the TechCrunch report on the Google antitrust case published on October 31, 2023, and the TechCrunch report on the class action antitrust suit against Google published on December 15, 2023?",
    "Does 'The Verge' article suggest that Valve is expanding its focus beyond games in its store, while 'Polygon' and 'Engadget' articles indicate that Valve is discontinuing the development and launch of new hardware, as seen with the cessation of updates to the Steam Deck and the absence of a Steam Deck OLED release?",
    "Does the TechCrunch article suggest that Sam Bankman-Fried directed Caroline Ellison to misuse customer funds, while the Fortune article claims that the entire success of FTX was based on lies, and does the second TechCrunch article allege that Sam Bankman-Fried's fraudulent actions were motivated by personal gain?",
    "After the report by The Age on October 22, 2023, suggesting the possibility of foul play on Google's part, did TechCrunch's stance on December 15, 2023, regarding Google's anticompetitive behavior towards news publishers show agreement or disagreement?",
    "Who is the individual associated with both FTX and Alameda Research, who faced legal scrutiny for their inability to manage significant growth and alleged fraudulent activities, as reported by The Verge and TechCrunch?",
    "Between the TechCrunch report on Sam Bankman-Fried's trial published on October 1, 2023, and the TechCrunch report on the prosecution's allegations against Sam Bankman-Fried published on October 7, 2023, was there consistency in the portrayal of Sam Bankman-Fried's legal challenges?",
    "Which company, recently reported by TechCrunch, has been involved in an antitrust court case providing extensive evidence against claims of hiding discovery items, has received early impressions for a product compared to OpenAI's GPT-3.5, and is accused by news publishers of harming their revenues through anticompetitive practices?",
    "Between the report from The Verge on Apple's defense of its Google Search deal published on September 26, 2023, and the TechCrunch report on the class action antitrust suit against Google published on December 15, 2023, was there a change in the portrayal of Google's market influence and competitive practices?",
    "Does the Sporting News article anticipate an impressive performance in the upcoming home game for Jordan Love, while the CBSSports.com article reports on Derrick Henry's performance in a recent game, specifically mentioning his two touchdowns and 76 rushing yards?",
    "Does the CBSSports.com article suggest that the Tampa Bay Buccaneers could not have success with a Michigan quarterback unlike past achievements, while the Sporting News article indicates that The Big Ten is currently engaged in a review process concerning Michigan and Jim Harbaugh, without implying any success with a quarterback?",
    "Who is the individual under 30, previously reported by TechCrunch as the richest person with philanthropic intentions, that is also the subject of allegations by the prosecution for committing fraud to gain wealth and influence, as covered by both TechCrunch and Fortune?"
]

def build_pipeline():
    """
    Initial components and indices for retriever
    """

    print("Initializing components...")

    analyser = QueryAnalyser()
    decomposer = QueryDecomposer()
    evaluator = RetrievalEvaluator()

    print("Loading ground truth...")
    ground_truth = evaluator.load_ground_truth(DATASET_PATH)

    print("Building retriever indices...")
    retrievers = dict()
    unique_embedders = set(analyser.strategies.values())

    for embedder in unique_embedders:
        retriever = Retriever(embedder=embedder, top_k=10)
        retriever.build_index(CORPUS_PATH)
        retrievers[embedder] = retriever

    return analyser, decomposer, retrievers, evaluator, ground_truth


def collect_pipeline_outputs(query, analyser, decomposer, retrievers, evaluator, ground_truth):
    """
    Run a target through the pipeline to collect contexts and ground_truth
    """

    analysis = analyser.analyse(query)
    retriever = retrievers[analysis['embedder']]

    retrieved_nodes = []

    if analysis['complexity'] in ['complex', 'very_complex']:
        subqueries = decomposer.decompose(query)

        for sq in subqueries:
            nodes = retriever.retrieve(sq)
            retrieved_nodes.extend(nodes)
    else:
        retrieved_nodes = retriever.retrieve(query)

    result = evaluator.evaluate(query, retrieved_nodes, ground_truth)

    contexts = [node.node.text for node in retrieved_nodes]

    # As RAGAS expects all expected evidence from ground_truth to be a single string
    expected_evidence = ground_truth.get(query, {}).get('evidence', [])
    ground_truth_str = " ".join(expected_evidence)

    return {
        'retrieved_contexts': contexts,
        'reference': ground_truth_str,
        **result
    }

def main():
    analyser, decomposer, retrievers, evaluator, ground_truth = build_pipeline()

    print(f"\nCollecting pipeline output for {len(TARGET_QUERIES)} targeted Population B queries...")

    samples = []
    results = []

    llm = llm_factory('gpt-4o-mini', client=AsyncOpenAI())
    scorer = ContextRecall(llm=llm)

    for i, query in enumerate(TARGET_QUERIES, 1):
        print(f"Query [{i} / {len(TARGET_QUERIES)}] Passing through pipeline...")
        sample = collect_pipeline_outputs(query, analyser, decomposer, retrievers, evaluator, ground_truth)

        print(f"Query [{i} / {len(TARGET_QUERIES)}] Scored through RAGAS...")
        score_result = scorer.score(
            user_input = sample['query'],
            retrieved_contexts = sample['retrieved_contexts'],
            reference = sample['reference']
        )

        results.append({
            'context_recall': score_result.value,  # Extract the score value
            **sample,
        })

    print("\n" + "="*70)
    print("RAGAS CONTEXT_RECALL RESULTS")
    print("="*70)

    # for i, result in enumerate(results):
    #     print(f"\nQuery {i}:")
    #     print(result)

    ragas_calibration_output_path = OUTPUT_DIR / f"ragas_calibration_output.json"

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(ragas_calibration_output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__=='__main__':
    main()