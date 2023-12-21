import os
from time import time
from pprint import pprint
from pathlib import Path
import pandas as pd
import streamlit as st
from getpass import getpass
import tiktoken

from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.callbacks import get_openai_callback

from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex, get_response_synthesizer
from llama_index.llms import OpenAI
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.storage import StorageContext
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.indices.loading import load_index_from_storage


st.title('Mineral Investment Pitch Analysis')

# add social links

# add description 

# add column for uploading pitch file
 
    # add code for openai api input
    os.environ['OPENAI_API_KEY'] = getpass() # enter your openai API key
    openai_llm = ChatOpenAI(
        model="gpt-3.5-turbo", # try gpt-4 if you want the best results
        temperature=0.1, # control how creative the ai can be   
    )
    file_dir = './'
    # add code for upload 
    llama_idx_docs = SimpleDirectoryReader(input_dir="./data").load_data()
    
    def build_sentence_window_index(
            documents,
            llm,
            embed_model="local:BAAI/bge-small-en-v1.5",
            save_dir="./sentence_index"
        ):
        # create the sentence window node parser w/ default settings
        print('building node parser')
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            )

        print('building service context')
        sentence_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            node_parser=node_parser,
            )

        if not os.path.exists(save_dir): # save the index to your directory to avoid making extra API calls (i.e, pay more) to openai
            print('building vector store')
            sentence_index = VectorStoreIndex.from_documents(
                documents,
                service_context=sentence_context
                )

            print('saving vector store')
            sentence_index.storage_context.persist(
                persist_dir=save_dir
            )

        else:
            print('loading vectore store')
            sentence_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=save_dir),
                service_context=sentence_context,
                )

        return sentence_index

    sentence_index = build_sentence_window_index(
            llama_idx_docs,
            openai_llm,
            embed_model="local:BAAI/bge-small-en-v1.5",
            save_dir="./sentence_index"
        )

    def get_sentence_window_query_engine(
            sentence_index,
            type='query', # or retrieve
            similarity_top_k=10, # play with this value to tweak how much data gets passed to the ai
            rerank_top_n=2
        ):

        # define postprocessors
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n,
            model="BAAI/bge-reranker-base"
            )

        if type == 'query': # for when we want to do queries and have AI synthesize a result
            sentence_window_engine = sentence_index.as_query_engine( # as_retriever( if you only want to do document retrieval
                similarity_top_k=similarity_top_k,
                node_postprocessors=[postproc, rerank],
                )

        elif type == 'retrieve': # for when we only want to get the documents back
            retriever = VectorIndexRetriever(
                index=sentence_index,
                similarity_top_k=10,

            )
            # assemble query engine with no_text response mode
            sentence_window_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=get_response_synthesizer(),
                node_postprocessors=[postproc, rerank],
            )

        else:
            raise ValueError(f'unknown retriever engine type requested: {type}')

        return sentence_window_engine
    # add code for displaying wait icon
    # add code for ingest status 

# add column for viewing results 
    # read the directory outputs
    print(len(llama_idx_docs), "\n") # get # of pages
    print(llama_idx_docs[0]) # show the first doc
    # add code for generating questions 
    get_questions_template = PromptTemplate.from_template(
        """
        You are an expert investor specializing in mineral mining investments.
        Your job is to create a comprehensive set of questions
        to be used for analyzing a potential investment. Use the context
        below to generate a list of questions that will allow
        the investor to make an informed yes/no decision as well as a
        counter-offer to the business seeking the investment.

        %CONTEXT
        {context}
        """
        )

    question_generator = get_questions_template | openai_llm | StrOutputParser()
    # with some work, you could use the query engine on the pitch deck to get more relevant context as well
    generated_questions = question_generator.invoke({"context":"This investment is for a gold mining project in Nome, Alaska."})
    generated_questions = generated_questions.split('\n')

    # put questions in a display output 
    for question in generated_questions:
        pprint(question)

    sentence_window_doc_retriever = get_sentence_window_query_engine(sentence_index, 'retrieve')

    def retrieve_docs(query):
        response = sentence_window_doc_retriever.query(query)
        text = [node.text for node in response.source_nodes]
        return '\n'.join(text)

    investment_question_prompt = PromptTemplate.from_template(
            """
            You are an expert investor specializing in mineral investments.
            Your job is to provide succinct answers to the question below
            given the context.

            %CONTEXT {context}

            %QUESTION {question}
            """
        )

    setup_and_retrieval = RunnableParallel(
        {"context": retrieve_docs, "question": RunnablePassthrough()}
    )

    investment_qa_chain = setup_and_retrieval | investment_question_prompt | openai_llm | StrOutputParser()

    print(f'the AI will be asked to answer {len(generated_questions)} questions.')
    answers = investment_qa_chain.batch(generated_questions)

    # let's review each answer next to its question
    responses = pd.DataFrame({'question':generated_questions, 'answer':answers,})

    # save to disk so we don't have to run again.
    responses.to_csv('./data/ai_investor_responses.csv', index=False)

    for _, qa_s in responses.iterrows():
        pprint(qa_s.loc['question'].strip())
        pprint(qa_s.loc['answer'].strip())
        print('\n')


    summary_template = PromptTemplate.from_template("""
    You are an expert venture capitalist specializing in the mineral mining industry.
    Review the following question and answer summaries and then carefully reason over
    them to answer the question.

    %SUMMARIES
    {summaries}

    %QUESTION
    {question}
    """)

    helper_chain = summary_template | openai_llm | StrOutputParser()

    summaries = ' '.join(['Question: ' + question + ' Answer: ' + answer for  idx, question, answer in responses.itertuples()])
    pprint(summaries[:500])

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
    num_tokens = len(encoding.encode(summaries))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

    print(f"using {num_tokens} tokens -> remaining tokens: {4096 - num_tokens}")



    with get_openai_callback() as cb:

        question = """what are the top 3-4 most important questions
                    to ask the entrepreneur about this pitch which are not
                    addressed in the summaries?"""

        answer = helper_chain.invoke({'summaries':summaries, 'question':question})

        print(cb, '\n\n')

    pprint(answer)


    with get_openai_callback() as cb:

        question = """There is never an idea investment decision where you have
        all the data. Knowing that, how would you evaluate the project? You MUST
        make a decision.

        Based on the limited data, would it make a good investment based on what
        you see in the summaries?

        Answer as a yes / no, and provide some context for how you would make
        the decision under uncertainty."""

        answer = helper_chain.invoke({'summaries':summaries, 'question':question})

        print(cb, '\n')

    pprint(answer)

    
    with get_openai_callback() as cb:

        question = """Based on the summaries, how could the investor improve
        their pitch deck for this investment opportunity?"""

        answer = helper_chain.invoke({'summaries':summaries, 'question':question})

        print(cb, '\n\n')

    pprint(answer)


    with get_openai_callback() as cb:

        question = """Based on the lack of information provided in this pitch deck,
                    and knowing that they're offering a 10% stake for $5M,
                    can you please produce a range of counteroffers for low-ball,
                    reasonable, and high-water offers?

                    If you don't have enough information, just throw a number out
                    there."""

        answer = helper_chain.invoke({'summaries':summaries, 'question':question})

        print(cb, '\n\n')

    pprint(answer)



    # add special questions to ask 


    # add record questions and pitches that're asked -> use zapier to send to google sheets