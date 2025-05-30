Procurement Multi Agent Streamlit code in Camel Roleplaying Framework

This Multi Agent Framework aims to automate and handles complex decision boundaries involved in each stage of Procurement Pipeline viz., 

**Product vendor reasearch and review**: Extensive resaerch over product supplier, reviews, delivery timelines, quality , cutomer care, compliance, regional blocks etc., and define metrics to 
evaluate and rank the vendor. 

Classification and Identification of RFP template: Different product have different RFP template and metrics, Classify product category and domain

RFP Generation : Generate RFP specific to request by incorporating all key user request and appropriate RFP template.

Validation: Review and Validate the RFP

For this purpose we developed a multi agent framework where respective agents handle specific tasks and collaborate with one another. 

Agent Architecture:

Orchestrator Agent to consolidate each agent's output and make the agents collaborate and make sure end result is achieved. 

Search Agent: Searches and reserch web extensively looking for webpages, forums, compliance research, regulations research, tax research, geo political research and ranks vendors

Classification agent: categorise and identifies domain and sub domain to extract appropriate RFP template

RFP gen agent: generates RFP with all key details

Validator agent: Based on pre defined business rules validates the RFP. 
