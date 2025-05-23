import streamlit as st
import openai
import os
import time
import random
import requests
import json
import textwrap


import nest_asyncio
nest_asyncio.apply()

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from camel.agents import ChatAgent
from camel.toolkits import SearchToolkit
# from camel.workforce import Workforce # We might not use Workforce directly for sequential calls
# from camel.tasks import Task        # We might not use Task directly for sequential calls

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

# --- LangChain Imports for Q&A ---
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')



OPENAI_API_KEY = config.get('API_KEYS', 'OPENAI_API_KEY')
PERPLEXITY_API_KEY = config.get('API_KEYS', 'PERPLEXITY_API_KEY')
llm_type = config.get('MODEL_CONFIG', 'LLM_TYPE')
model_name = config.get('MODEL_CONFIG', 'MODEL_NAME')
PERPLEXITY_API_URL = config.get('MODEL_CONFIG', 'PERPLEXITY_API_URL')
NOT_FOUND_IN_CONTEXT_PHRASE=config.get('PROMPTS_GENERAL','NOT_FOUND_IN_CONTEXT_PHRASE')
PERPLEXITY_MINIMAL_SYSTEM_PROMPT=config.get('PROMPTS_GENERAL','PERPLEXITY_MINIMAL_SYSTEM_PROMPT')
ADVANCED_SEARCH_PROMPT_TEMPLATE=config.get('PROMPTS_SEARCH_AGENT','ADVANCED_SEARCH_PROMPT_TEMPLATE')
SEARCH_AGENT_SYSTEM_MESSAGE=config.get('PROMPTS_SEARCH_AGENT','SEARCH_AGENT_SYSTEM_MESSAGE')
CLASSIFICATION_SYSTEM_MESSAGE=config.get('PROMPTS_CLASSIFICATION_AGENT','CLASSIFICATION_SYSTEM_MESSAGE')
ADVANCED_RFP_GENERATION_MESSAGE=config.get('PROMPTS_RFP_AGENT','ADVANCED_RFP_GENERATION_MESSAGE')
APPROVAL_AGENT_SYSTEM_MESSAGE=config.get('PROMPTS_APPROVAL_AGENT','APPROVAL_AGENT_SYSTEM_MESSAGE')




class CamelUtils:

  
  
    def make_chat_agent(self,role_name: str, system_content: str, model_instance) -> ChatAgent:
        sys_msg = BaseMessage.make_assistant_message(role_name=role_name, content=system_content)
        return ChatAgent(system_message=sys_msg, model=model_instance)
    

   
    

    def create_search_agents(self, shared_llm_model, rfp_llm_model):
        search_agent = self.make_chat_agent(
            "Strategic Sourcing Analyst", SEARCH_AGENT_SYSTEM_MESSAGE, shared_llm_model
        )
        
        return search_agent
    
    def create_classification_agents(self, shared_llm_model, rfp_llm_model):
        classification_agent = self.make_chat_agent(
            "Product Classification Specialist", CLASSIFICATION_SYSTEM_MESSAGE, shared_llm_model
        )
        
        return classification_agent
    
    def create_rfp_agents(self, shared_llm_model, rfp_llm_model):
        rfp_agent = self.make_chat_agent(
            "RFP Generation Specialist", ADVANCED_RFP_GENERATION_MESSAGE, rfp_llm_model
        )
        
        return rfp_agent
    
    def create_approval_agents(self, shared_llm_model, rfp_llm_model):
        approval_agent = self.make_chat_agent(
            "Procurement Approval Officer", APPROVAL_AGENT_SYSTEM_MESSAGE, shared_llm_model
        )
        
        return approval_agent
    

    def query_perplexity_sonar(self,detailed_user_request_prompt, model_name="sonar"):
        """
    Queries the Perplexity API with a detailed user request prompt,
    including enhanced error handling and debugging.
    """
        if not PERPLEXITY_API_KEY : # Basic check
            st.error("🚨 Perplexity API Key is not configured correctly!")
            return "Perplexity API Key not configured."

        if not PERPLEXITY_API_URL:
            st.error("🚨 Perplexity API URL is not configured!")
            return "Perplexity API URL not configured."

        headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json" # Often good to include Accept header
        }
        payload = {
            "model": model_name, # Make sure this model name is correct for your API access
            "messages": [
            {
                "role": "system",
                "content": PERPLEXITY_MINIMAL_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": detailed_user_request_prompt
            }
        ],
        "temperature": 0.3, # Adjust as needed
        # "max_tokens": 2048, # If the API supports it and you need longer responses
        # "stream": False, # Explicitly set to False unless you handle streaming
        }

        #st.info("Constructed Perplexity API Request:")
        print("Constructed Perplexity API Request:")
   
        print({"url": PERPLEXITY_API_URL, "payload": payload})

        api_response_text = "No response yet or error before request." # Default message

        try:
            st.write("Attempting to send request to Search Agent...")
            print("Attempting to send request to Search Agent...")
            # Increased timeout: requests library default is no timeout, but good practice for external APIs
            # Perplexity might have its own server-side timeouts.
            response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=120) # 120 seconds timeout
            #st.write(f"Received response status code: {response.status_code}")
            print(f"Received response status code: {response.status_code}")
            api_response_text = response.text # Store raw text for debugging in case of JSON decode error

            response.raise_for_status()  # This will raise an HTTPError for 4xx/5xx responses

            api_response_data = response.json() # Try to parse JSON

            st.info("Successfully received and parsed JSON response from Perplexity API:")
            print("Successfully received and parsed JSON response from Perplexity API:")
            #st.json(api_response_data) # Display the full successful JSON response
            print(api_response_data)


       
            if "choices" in api_response_data and isinstance(api_response_data["choices"], list) and api_response_data["choices"]:
                first_choice = api_response_data["choices"][0]
                if "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in first_choice["message"]:
                    raw_search_output = first_choice["message"]["content"]
                    if raw_search_output:
                        return raw_search_output.strip()
                    else:
                        st.warning("Perplexity API returned an empty content string.")
                        print("Perplexity API returned an empty content string.")
                        return "Perplexity API returned empty content."
                else:
                    st.warning("Perplexity API 'message' or 'content' field missing or not in expected format in the first choice.")
                    print("Perplexity API 'message' or 'content' field missing or not in expected format in the first choice.")
                    return "Perplexity API response 'message' or 'content' format unexpected."
            else:
                st.warning("Perplexity API response 'choices' field missing, not a list, or empty.")
                print("Perplexity API response 'choices' field missing, not a list, or empty.")
                return "Perplexity API response 'choices' format unexpected."

        except requests.exceptions.Timeout:
            st.error(f"Perplexity API request timed out after 120 seconds.")
            return "API request timed out."
        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"Perplexity API connection error: {conn_err}")
            return f"API connection error: {conn_err}"
        except requests.exceptions.HTTPError as http_err:
            st.error(f"Perplexity API HTTP error: {http_err}")
            st.error(f"Response Status: {http_err.response.status_code}")
            st.error(f"Response Body: {api_response_text}") # Show raw response text on HTTP error
            print(f"Response Body: {api_response_text}")
            return f"API HTTP error {http_err.response.status_code}."
        except json.JSONDecodeError:
            st.error("Failed to decode JSON response from Perplexity API.")
            st.error(f"Raw Response Text that failed to parse: {api_response_text}")
            print(f"Raw Response Text that failed to parse: {api_response_text}")
            return "API response was not valid JSON."
        except Exception as e:
            st.error(f"An unexpected error occurred while calling Perplexity API: {e}")
            print(f"An unexpected error occurred while calling Perplexity API: {e}")
            st.error(f"Raw Response Text (if available): {api_response_text}")
            import traceback
            st.text(traceback.format_exc())
            return f"Unexpected API error: {e}"


    def create_pdf_from_markdown(self,markdown_text, filename="generated_rfp.pdf"):
    # ... (Your full implementation) ...
        buffer = BytesIO(); doc = SimpleDocTemplate(buffer, pagesize=letter); styles = getSampleStyleSheet(); story = []
        lines = markdown_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("### "): story.append(Paragraph(line[4:], styles['h3']))
            elif line.startswith("## "): story.append(Paragraph(line[3:], styles['h2']))
            elif line.startswith("# "): story.append(Paragraph(line[2:], styles['h1']))
            elif line.startswith("- ") or line.startswith("* "):
                if 'BulletText' not in styles: styles.add(ParagraphStyle(name='BulletText', parent=styles['Normal'], leftIndent=0.25*inch, bulletIndent=0.1*inch))
                story.append(Paragraph(line[2:], styles['BulletText'], bulletText='•'))
            elif line: story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        try: doc.build(story); buffer.seek(0); return buffer
        except Exception as e: st.error(f"PDF Error: {e}"); return None


    # --- Agent Trigger Functions ---
    def trigger_search_agent_processing(self,user_query_details_dict,shared_llm_model,rp_llm_model):
        st.info("🤖 Search Agent  processing...")
    # The ADVANCED_SEARCH_PROMPT_TEMPLATE is now the user message to AthenaSource
    # AthenaSource's system message already defines its role and how it should behave.
        formatted_user_prompt_for_search = ADVANCED_SEARCH_PROMPT_TEMPLATE.format(**user_query_details_dict)
        with st.expander("Prompt to Search Agent"):
            st.text_area("Prompt:", value=formatted_user_prompt_for_search, height=300, disabled=True)

        with st.spinner("SearchAgent is conducting research..."):
        # In this model, AthenaSource would be prompted to generate search queries
        # for Perplexity, or more simply, it's prompted to act AS IF it has done
        # the research using the capabilities described in its system message.
        # For a true Perplexity call, you'd do it here and feed results to an *analyzer* agent.
        # For this demo, we assume search_agent uses its LLM to synthesize what Perplexity *would* find.
        # To actually use Perplexity, the search_agent would need a tool or this function would call it.

        # Let's refine: This agent will *process* the Perplexity output
        # Step 1: Call Perplexity
            perplexity_raw_output = self.query_perplexity_sonar(formatted_user_prompt_for_search) # Pass the structured prompt
            st.session_state.raw_perplexity_output = perplexity_raw_output # Store for debugging

            if "Error:" in perplexity_raw_output or "PPLX:" in perplexity_raw_output and "No content" in perplexity_raw_output :
                 st.session_state.search_results = f"Perplexity search failed or returned no results: {perplexity_raw_output}"
                 st.error(st.session_state.search_results)
                 return

        # Step 2: AthenaSource (Search Agent) analyzes Perplexity's output
        # The system message for search_agent already tells it to analyze, rank, and output top vendors.
        # We feed it the raw Perplexity output and ask it to perform its role.
        
            analysis_prompt_for_search_agent = textwrap.dedent(f"""
            You have received the following raw research data from an extensive web search (simulated via Perplexity API):
        --- RAW SEARCH DATA START ---
            {perplexity_raw_output}
        --- RAW SEARCH DATA END ---

        Based on your role as AthenaSource, a Strategic Sourcing Analyst, and the original procurement request details embedded in the initial research prompt you received, please now:
        1. Analyze this raw search data.
        2. Identify potential vendors.
        3. Rank them based on criteria like capability, reputation, regional presence, and initial risk assessment.
        4. Filter and output the top 3-5 vendors with summarized findings and justifications for their ranking, adhering to your persona and output format.
        """)
            # The search_agent's system message already defines its persona and output format.
            # We are giving it a user message containing the data to process.
            search_agent=self.create_search_agents(shared_llm_model,rp_llm_model)
            response_msg = search_agent.step(
                BaseMessage.make_user_message(role_name="Orchestrator", content=analysis_prompt_for_search_agent)
            )
            if response_msg:
                st.session_state.search_results = response_msg.msgs[0].content
                st.success("✅ Search Agent analysis complete!")
            else:
                st.session_state.search_results = "Search Agent failed to provide analysis."
                st.error(st.session_state.search_results)

    def trigger_classification_agent_processing(self,user_query, search_results_text,shared_llm_model,rp_llm_model):
        st.info("🤖 Classification Agent  processing...")
        prompt_for_classification = f"User Procurement Request:\n{user_query}\n\nSearch Agent Findings:\n{search_results_text}"
        classification_agent=self.create_classification_agents(shared_llm_model,rp_llm_model)
        with st.spinner("Agent is classifying..."):
            response_msg = classification_agent.step(
            BaseMessage.make_user_message(role_name="Orchestrator", content=prompt_for_classification)
        )
        if response_msg:
            st.session_state.classification_output = response_msg.msgs[0].content
            st.success("✅ Classification Agent complete!")
        else:
            st.session_state.classification_output = "Classification Agent failed."
            st.error(st.session_state.classification_output)


    def trigger_rfp_agent_processing(self,user_query, classification_output_text, search_results_text,shared_llm_model,rp_llm_model):
        st.info("🤖 RFP Generation Agent processing...")
        rfp_context = f"""
        **Original User Procurement Need:**
        {user_query}

        **Procurement Category Determined:**
        {classification_output_text}

        **Key Findings from Initial Vendor/Solution Search (for context):**
        {search_results_text[:1500]}
        """ # Pass context to the RFP agent's system prompt
        rfp_agent=self.create_rfp_agents(shared_llm_model,rp_llm_model)
        with st.spinner("RFP Agent is drafting the RFP..."):
        # The ADVANCED_RFP_GENERATION_MESSAGE_FOR_AGENT_DEF is its system message.
        # The user message here provides the specific context for *this* RFP.
            response_msg = rfp_agent.step(
            BaseMessage.make_user_message(role_name="Orchestrator", content=rfp_context)
        )
        if response_msg:
            st.session_state.rfp_content_markdown = response_msg.msgs[0].content
            st.success("✅ RFP Generation Agent complete!")
        else:
            st.session_state.rfp_content_markdown = "RFP Generation Agent failed."
            st.error(st.session_state.rfp_content_markdown)


    def trigger_approval_agent_processing(self,user_query, search_results_text, classification_text, rfp_text,shared_llm_model,rp_llm_model):
        st.info("🤖 Approval Officer processing...")
        approval_context = f"""
    **Original User Procurement Need:** {user_query}
    **Search Agent Findings:**\n{search_results_text[:1000]}
    **Classification Output:**\n{classification_text}
    **Generated RFP Draft:**\n{rfp_text[:1500]}

    Please evaluate this procurement package and make an approval decision with justification.
    """
        approval_agent=self.create_approval_agents(shared_llm_model,rp_llm_model)
        with st.spinner("Approval Agent is reviewing for approval..."):
            response_msg = approval_agent.step(
             BaseMessage.make_user_message(role_name="Orchestrator", content=approval_context)
        )
        if response_msg:
            st.session_state.approval_output = response_msg.msgs[0].content
            st.success("✅ Approval Officer complete!")
        else:
            st.session_state.approval_output = "Approval Agent failed."
            st.error(st.session_state.approval_output)

    @staticmethod
    @st.cache_resource
    def get_qna_chain_for_combined_context():
        llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
        template_str = f"""
        You are a helpful AI assistant for procurement analysis... (Your full Q&A prompt from before,
        instructing to use search_results_context AND rfp_document_context, and to use {NOT_FOUND_IN_CONTEXT_PHRASE})

        Initial Search Results:
        -----------------------
        {{search_results_context}}
        -----------------------

        Generated RFP Document:
        -----------------------
        {{rfp_document_context}}
        -----------------------

        User Question: {{user_question}}
        Answer:
        """
        prompt = PromptTemplate(
            input_variables=["search_results_context", "rfp_document_context", "user_question"],
            template=template_str
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    def answer_question_orchestrator(self,chain: object,user_question: str, search_results: str, rfp_markdown: str, perplexity_model_name: str = "sonar"):
        search_context = search_results if search_results else "No initial search results available."
        rfp_context = rfp_markdown if rfp_markdown else "No RFP document generated yet."
        context_based_answer = chain.run(
        search_results_context=search_context,
        rfp_document_context=rfp_context,
        user_question=user_question
        )
        if NOT_FOUND_IN_CONTEXT_PHRASE.strip().lower() in context_based_answer.strip().lower():
            perplexity_query = f"Regarding a procurement process, please find information for: {user_question}"
            fallback_answer = query_perplexity_sonar(query_prompt=perplexity_query, model_name=perplexity_model_name, is_follow_up=True)
            if fallback_answer and not any(err_msg.lower() in fallback_answer.lower() for err_msg in ["error:", "unexpected", "not configured", "timed out"]):
                return f"Info not in current docs. New search found:\n\n{fallback_answer}"
            else:
                return f"Info not in current docs. Fallback search also failed: {fallback_answer}"
        return context_based_answer


