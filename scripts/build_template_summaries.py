# build_template_summaries.py
import pickle
import argparse
from pathlib import Path
from collections import Counter


# -------------------------------------------------------------------
# STEP 1: Your exact mapping – Template # → Summary
# I'm using the exact text you provided for each template.
# -------------------------------------------------------------------

TEMPLATE_SUMMARIES = {
    # Template #1: Irrelevant (0.70x). Rank 37.
    "#1": "Current claim: managing mid-market enterprise SaaS software quota metrics with zero hands-on software development or machine learning platform exposure.",
    # Template #2: Irrelevant (0.70x). Rank 34.
    "#2": "Current claim: acting as an IT services business analyst with limited technical depth in production AI modeling frameworks.",
    # Template #3: Irrelevant (0.70x). Rank 41.
    "#3": "Current claim: handling general ledger transactions, statutory audits, and tax compliance with no software engineering capabilities.",
    # Template #4: Irrelevant (0.70x). Rank 38.
    "#4": "Current claim: directing account-based marketing and lead generation strategies for business development with zero engineering footprints.",
    # Template #5: Irrelevant (0.70x). Rank 42.
    "#5": "Current claim: drafting publication SEO marketing copy and utilizing consumer LLM tools for superficial text updates.",
    # Template #6: Irrelevant (0.70x). Rank 40.
    "#6": "Current claim: monitoring physical warehouse supply chain metrics, fulfillment KPIs, and shift workforce productivity levels.",
    # Template #7: Irrelevant (0.70x). Rank 43.
    "#7": "Current claim: designing visual asset layouts, packaging configurations, and graphic brand guidelines using Figma tools.",
    # Template #8: Irrelevant (0.70x). Rank 39.
    "#8": "Current claim: supervising tier-1 customer ticketing workflows and platform escalations with zero production backend system experience.",
    # Template #9: Irrelevant (0.70x). Rank 44.
    "#9": "Current claim: configuring hardware CAD designs, finite element analysis modeling, and factory physical tooling setups.",
    # Template #10: Irrelevant (0.70x). Rank 36.
    "#10": "Current claim: engineering client-side native mobile experiences for Android applications using Java and Kotlin.",
    # Template #11: Irrelevant (0.70x). Rank 29.
    "#11": "Current claim: operating cloud infrastructure networks, CI/CD automated test runs, and platform clusters using AWS and Terraform.",
    # Template #12: Irrelevant (0.70x). Rank 30.
    "#12": "Current claim: building web software products and database schemas with Node, React, and Postgres without statistical modeling exposure.",
    # Template #13: Irrelevant (0.70x). Rank 32.
    "#13": "Current claim: crafting user interface components, design systems, and client accessibility flows using React and TypeScript.",
    # Template #14: Irrelevant (0.70x). Rank 35.
    "#14": "Current claim: executing manual interface validation and end-to-end regression testing suites using Selenium and pytest.",
    # Template #15: Irrelevant (0.70x). Rank 31.
    "#15": "Current claim: developing enterprise transactional software infrastructure and microservice APIs with Java and Spring Boot.",
    # Template #16: Data Engineers (0.95x). Rank 26.
    "#16": "Current claim: structuring data warehouses, lake houses, and metadata catalogs at early-stage product startups.",
    # Template #17: Data Engineers (0.95x). Rank 28.
    "#17": "Current claim: transformation pipeline engineering, cleaning messy datasets, and configuring analytics views via dbt and Snowflake.",
    # Template #18: Irrelevant (0.70x). Rank 25.
    "#18": "Current claim: scaling high-throughput Python backend wrappers and observability layers around pre-existing predictive endpoints.",
    # Template #19: Data Engineers (0.95x). Rank 24.  # <-- NEWLY ADDED
    "#19": "Current claim: splitting time between lightweight ML modeling (clustering/classification) and analytics-engineering, with a focus on A/B testing frameworks – not a full-time ML deployment role.",
    # Template #20: Data Engineers (0.95x). Rank 27.
    "#20": "Current claim: building large-scale batch processing data lake engines using Apache Spark and transactional data storage layers.",
    # Template #21: Data Engineers (0.95x). Rank 21.
    "#21": "Current claim: deploying low-latency event streaming architecture and data bus topologies with Apache Kafka for downstream consumption.",
    # Template #22: ML-Adjacent (1.05x). Rank 17.
    "#22": "Current claim: implementing real-time streaming risk APIs and managing structured offline feature stores for fraud detection systems.",
    # Template #23: ML-Adjacent (1.05x). Rank 22.
    "#23": "Current claim: researching statistical filtering layers and building offline mathematical engines for matrix factorization.",
    # Template #24: ML-Adjacent (1.05x). Rank 23.
    "#24": "Current claim: training recursive time-series models and tree architectures for structural product demand forecasting operations.",
    # Template #25: Irrelevant (0.70x). Rank 33.
    "#25": "Current claim: exclusively building computer vision models for image moderation using PyTorch with zero professional NLP or information retrieval experience.",
    # Template #26: ML-Adjacent (1.05x). Rank 19.
    "#26": "Current claim: configuring tokenization strategies, building text preprocessing layers, and training custom transformer sequences.",
    # Template #27: ML-Adjacent (1.05x). Rank 20.
    "#27": "Current claim: optimizing business metric forecasting targets and tabular feature extraction flows using scikit-learn frameworks.",
    # Template #28: Core IR (1.30x). Rank 7.
    "#28": "Current claim: tuning search engine scoring logic, adjusting index configurations, and scaling product search relevancy features.",
    # Template #29: Core IR (1.30x). Rank 13.
    "#29": "Current claim: building behavior-driven ranking architectures and shipping gradient-boosted feed optimization models in production.",
    # Template #30: Core IR (1.30x). Rank 8.
    "#30": "Current claim: designing vector-based search infrastructure, embedding generation modules, and nearest-neighbor indices using FAISS.",
    # Template #31: ML-Adjacent (1.05x). Rank 16.
    "#31": "Current claim: engineering conversational RAG platforms, context window assembly layers, and prompt optimization middleware protocols.",
    # Template #32: Core IR (1.30x). Rank 14.
    "#32": "Current claim: developing candidate recommendation feeds and personalized candidate-to-recruiter matching systems at high scale.",
    # Template #33: ML-Adjacent (1.05x). Rank 18.
    "#33": "Current claim: managing end-to-end model registry networks, deployment orchestration pipelines, and automated artifact version tracking.",
    # Template #34: Core IR (1.30x). Rank 4.
    "#34": "Current claim: building neural ranking pipelines, dense information retrieval layers, and vector search indices using Pinecone.",
    # Template #35: Core IR (1.30x). Rank 9.
    "#35": "Current claim: engineering content discovery exploration tools and balancing item cold-start discovery using multi-armed bandits.",
    # Template #36: Core IR (1.30x). Rank 1.
    "#36": "Current claim: building hybrid search systems, dense/sparse retrieval layers, and configuring offline evaluation frameworks.",
    # Template #37: Core IR (1.30x). Rank 3.
    "#37": "Current claim: fine-tuning large open-source language models via parameter-efficient methods for structural candidate parsing.",
    # Template #38: Core IR (1.30x). Rank 11.
    "#38": "Current claim: architecting text query parsing tools, intent extraction microservices, and indexing topologies at scale-ups.",
    # Template #39: Core IR (1.30x). Rank 10.
    "#39": "Current claim: redesigning transactional match logic and replacing rule systems with ML-driven evaluation scoring blocks.",
    # Template #40: Core IR (1.30x). Rank 5.
    "#40": "Current claim: migrating standard keyword database configurations into multi-stage semantic lookup platforms.",
    # Template #41: Core IR (1.30x). Rank 12.
    "#41": "Current claim: optimizing operational ranking functions and balancing data lifecycle streams with strict online system checks.",
    # Template #42: Core IR (1.30x). Rank 2.
    "#42": "Current claim: scaling high-volume text embedding generation scripts and managing live vector indexing systems in production.",
    # Template #43 / Unique #1: Core IR (1.30x). Rank 6.
    "#43": "Current claim: supervising search and discovery metrics while aligning offline algorithmic test suites with production key metrics.",
    # Template #44 / Unique #2: Core IR (1.30x). Rank 15.
    "#44": "Current claim: managing personalization feature stores, inference data telemetry, and automated concept drift monitoring modules.",
}

# -------------------------------------------------------------------
# STEP 2: Load the pickle and map each fingerprint to its summary
# -------------------------------------------------------------------

def get_template_number(fingerprint, sorted_templates):
    """Returns the template number (#1 to #44) based on its position in the sorted list."""
    for idx, (fp, freq) in enumerate(sorted_templates, start=1):
        if fp == fingerprint:
            return f"#{idx}"
    return None

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts", help="Path to artifacts folder")
    parser.add_argument("--input", default="jd_templates.pkl", help="Input pickle filename (default: jd_templates.pkl)")
    parser.add_argument("--output", default="jd_templates_enhanced.pkl", help="Output pickle filename (default: jd_templates_enhanced.pkl)")
    args = parser.parse_args()

    artifacts_path = Path(args.artifacts)
    input_path = artifacts_path / args.input
    output_path = artifacts_path / args.output
    # Load the existing pickle
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    template_counter = data['template_counter']
    
    # Sort templates by frequency descending (most common first)
    sorted_templates = sorted(template_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Build the final summary map
    summary_map = {}
    for fingerprint, freq in sorted_templates:
        template_num = get_template_number(fingerprint, sorted_templates)
        if template_num and template_num in TEMPLATE_SUMMARIES:
            summary_map[fingerprint] = TEMPLATE_SUMMARIES[template_num]
        else:
            summary_map[fingerprint] = "Uncategorized template."

    # Add the new key to the data
    data['template_summaries'] = summary_map

    # Save the enhanced pickle
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"[build] Enhanced pickle saved to {output_path}")
    #Quick verification: print the first 5 mappings
    print("Sample mappings (Template #, Summary):")
    for i, (fp, summary) in enumerate(list(summary_map.items())[:5]):
        template_num = get_template_number(fp, sorted_templates)
        print(f"  {template_num}: {summary[:80]}...")

if __name__ == "__main__":
    main()