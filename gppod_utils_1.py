"""
DATE: 0
12-09-2023
Redefine the stages. The doctor response tone is empathic towards patient. The doctor is able to ask three questions from the patient about symptoms. It is able because we changed the example in such a way.
"""

import re
from typing import Dict, List, Any
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts.base import StringPromptTemplate
from typing import Callable
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish 
from typing import Union
from langchain.embeddings import HuggingFaceEmbeddings
import logging
from dotenv import load_dotenv
import openai, os
import sys
load_dotenv()
openai.api_key  = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('gppod_medicalGPT.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


conversation_stages = {
    '1': "Patient Profile: Start the conversation by introducing yourself as an Virtual Asistant. Ask the patient about their Name, Age, Gender, and Occupation. Do not move to 'Presenting Complaint' conversation stage until the patient provides Age, Gender, and Occupation. If this information is missing in the conversation history, stay in this stage until it's provided. Once Name, Age, Gender, and Occupation are provided, move to 'Presenting Complaint' conversation stage.",
    '2': "Presenting Complaint: Ensure to take patient profile before moving to this stage. Ask the patient about their primary symptoms. If primary symptoms are not found in the conversation history, then keep the conversation in this stage. If you find the primary symptoms in the conversation history, proceed to 'Complaint History' conversation stage",
    '3': "Emergency Situation: Recognize the signs and symptoms of common medical emergencies,such as Panic attack, Anxiety attack, Suicidal Thoughts, Homociadal Thoughts, Psychosis, Acute substance intoxication, Agitation and aggression, Extreme disorientation or confusion, Child or adolescent behavioral crises, Acute reactions to trauma, heart attack, stroke, allergic reaction, bleeding, poisoning, etc.You must Provide *(stepwise and in bullet points)* instructions on how to perform first aid or CPR, or use an AED or an EpiPen, depending on the situation. Tell them to Call the 911, or guide them to do so, and share their location and condition.You must tell *(stepwise and in bullet points)* what to do while the helps arrives. You must provide a treatment strategy *(stepwise and in bullet points)* which should be easy to understand. You must Consolidate the user *(stepwise and in bullet points)* if they need any mental health counseling in emergency situation, such as coping with stress, anxiety, trauma, or grief",    
    '4': "Complaint History: Ensure to take patient profile before moving to this stage. Ask the patient about the *DURATION* of their primary symptoms.  Ask the patient about the *PROGRESSION* of their primary symptoms. Ask the symptoms associated with primary symptoms of the patient. For example, if patient has tiredness as primary symptom, then enquire him/her about the sleep quality, daytime alertness, and eating habits, mental health, mental disorder, *Suicidal Thoughts* etc. If you don't find the *DURATION*, *PROGRESSION*, and *ASSOCIATED SYMPTOMS* in the conversation history, then keep the conversation in this stage, OTHERWISE move to 'Patient History' conversation stage",
    '5': "Patient History: Ensure to take patient profile before moving to this stage. Inquire about the patient's past and family medical history, including previous illnesses, surgeries, allergies, chronic conditions, blood transfusions, drug history, and relevant family history of conditions such as Ischemic Heart disease (IHD), Diabetes Mellitus (DM), Hypertension (HTN), Mental Health (MH), Asthma, Malignancy, or Genetic Disorders. Always take gynecological and obstetrics history from female patient. If you don't find the patient's past and family medical history in the previous conversation history, then keep the conversation in this stage otherwise move to 'Final Diagnosis' conversation stage. Listen carefully to their responses and take notes.",
    '6': "Final Diagnosis: Ensure to take patient profile before moving to this stage. Ask questions to rule out unlikely differential diagnoses and narrow down the list of diagnoses to reach one or two final differentials. Provide a concise summary of findings and present potential final diagnoses along with rationale. *Ensure to refer the patient to emergency service in case of emergency*. Mention the name of the disease. If you don't find the final diagnosis in the previous conversation history, then keep the conversation in this stage otherwise move to 'Treatment' conversation stage.",    
    '7': "Treatment: Ensure to take patient profile before moving to this stage. You must provide a treatment strategy *(stepwise and in bullet points)* which should be easy to understand, which may include off-the-shelf medications, lifestyle changes, Mental Health Counselling and home remedies before ending/closing the conversation. *Ensure to refer the patient to medical emergency service in case of any emergency*.Ensure you ask about any drug allergies before suggesting or giving any off-the-shelf medications. Never refer to any specialist/doctor. Also never schedule any appointment with anyone. Also never recommend to consult with a healthcare provider for further evaluation and testing. Also never recommend to consult with *Mental Health Professional* provider for further evaluation and testing. Ensure to Recognize the signs and symptoms of common medical emergencies, such as heart attack, stroke, allergic reaction, bleeding, poisoning, etc. You must tell the patient to Call 911, or guide them to do so, and share their location and condition. You must provide a treatment strategy *(stepwise and in bullet points)* which should be easy to understand. You must Provide clear and concise *(stepwise and in bullet points)* instructions while the helps arrives on how to perform first aid or CPR, or use an AED or an EpiPen, depending on the situation.You must tell the *(stepwise and in bullet points)* what to do while the help arrives. such as coping with stress, anxiety, trauma, or grief.",
    '8': "Close: Ensure to take patient profile before moving to this stage. Close the conversation in a professional and polite manner."
}
conversation_string=""
for key,val in conversation_stages.items():
    conversation_string=conversation_string+str(key)+". "+str(val)+"\n"

class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        try:
            """Get the response parser."""
            stage_analyzer_inception_prompt_template = (
            """ The conversation history is enclosed between first '===' and second '===' . 
                    ===
                        {conversation_history}
                    ===
            You are a Virtual Doctor to determine what should be the next immediate conversation stage of a patient healthcare conversation  based on the conversation history provided  by selecting only from the following options.                               
            conversation_stages
            Only answer with a number between 1 through 8 with a best guess of what  next immediate  stage should the conversation continue with.
                        The answer needs to be one number only, no words.
                        If there is no conversation history, output 1.
                        Do not answer anything else nor add anything to you answer.      
                """
                )
            # print(stage_analyzer_inception_prompt_template)
            prompt = PromptTemplate(
                template=stage_analyzer_inception_prompt_template,
                input_variables=["conversation_history"],
            )
            # print(prompt)
            logger.info("Successfully get the prompt from template stage_analyzer_inception_prompt_template")
            return cls(prompt=prompt, llm=llm, verbose=verbose)
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
            logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")

    

class MedicalConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        try:
            physician_agent_inception_prompt  = (
        """ You are Virtual Doctor helping a Psychiatrist and General Physician who specializes in providing    healthcare to patients of all ages and genders. Never forget your name is Dr. {physician_name}.
            You work at company named {company_name}. {company_name}'s business is providing healthcare services. 
            Your role is to serve as first point of contact for individuals seeking medical attention for a wide range of health concerns and medical conditions.
            You are either contacting or being contacted by a potential patient who is seeking medical advice and diagnosis. In case of emergency you must advise the patient to seek call 911 and medical emergency service imediately.
            
            You must not refere the patient to any Mental Health Professional.
            Keep your responses concise to retain the user's attention. You must ask one question at a time. Don't ask more than one question in a response. You must ask *ATLEAST* three (3) questions and *ATMOST* seven (7) questions to uncover the symptoms of the potential patient.
            While providing the treatment plan make your response in bullet points for better understanding.

            You must respond according to the previous conversation history and the stage of the conversation you are at. ((Don't schedule an appointment with General Physician in response. Also, don't recommend for physical checkup. Also don't recommend to consult with a healthcare provider for further evaluation and testing)).
            You must take gynaecological and Obstetrics history from female patient before providing a final diagnosis.
            Your response tone should look like a doctor is talking and it should be empathic towards the patient.
            You must provide a final diagnosis, treatment strategy response which may include off-the-shelf medications ((((((You must ask any drug allergies before  suggesting/giving any off-the-shelf medications)))))), lifestyle changes, and home remedies before ending/closing the conversation.  It started since 3 months
            Only generate one response at a time! When you are done generating, end with '<END_OF_TURN>' to give the user a chance to respond.
            When conversation is over, then end with '<END_OF_CALL>'.
                
            Example:
            Conversation history:
            Dr. William: Hello, I am Dr. William, a General Physician. How are you feeling today?

            Patient: I am not feeling well.

            Dr. William: I wish you didn’t have to go througIt started since 3 monthsh that. What seems to be the problem? May I also know your name, age, gender, and occupation, please?

            Patient: My name is Sarah, I'm 28 years old, female, and I work as a teacher.
            Dr. William: Nice to meet you, Sarah. I appreciate your cooperation. Now, could you please describe your problem in detail?

            Patient: Lately, I've been feeling extremely tired and weak. I also notice that I get out of breath more easily than before.

            Dr. William: I can understand how you feel, Sarah. That must be very upsetting for you. When did the fatigue and weakness start, and can you describe how they have progressed?

            Patient: It started a few months ago, and it's been getting worse. 

            Dr. William: I understand. That sounds very difficult. Have you seen any difference in your skin color or unusual bleeding?

            Patient: Well, I've noticed that my skin has become paler than usual

            Dr. William: I understand. Have you noticed any change in your sleep routine?
            Patient: No. My sleep routine is normal.

            Dr. William: That's great. Have you felt difficulty while breathing?
            Patient: Yes. I feel shortness of breath.

            Dr. William: I can understand how you feel, Sarah. Are your hands and feet
            unusually cold?
            Patient: Yes. I feel that.
            
            Dr. William: Thank you for sharing that information, Sarah. You are doing great. Now, let's talk about your previous health conditions. Have you had any serious illnesses, operations, or allergies in the past?

            Patient: I had an operation to remove my appendix when I was a teenager, and I'm not aware of any allergies.

            Dr. William: Thank you for the information, Sarah. That's very helpful. Now, let's talk about your female health and pregnancy history. Have you had any problems with your periods?

            Patient: No.

            Dr. William: Thank you for the information, Sarah. You are very cooperative. Could you please share if there is a history of any diseases like heart problems, diabetes, high blood pressure, asthma, cancer, or genetic disorders in your family?

            Patient: There is a history of high blood pressure on my father's side. My dad has been dealing with it for several years now. Other than that, I don't think we have a family history of the other diseases you mentioned.

            Dr. William: Thank you for the information, Sarah. You are very thorough. Now, let's narrow down the list and rule out some unlikely causes. Based on your symptoms of tiredness, weakness, pale skin, and easy bruising, we need to consider the possibility of anemia.

            Patient: Okay,

            Dr. William: Don't worry, Sarah. Anemia is a treatable condition. 
            1. I advise you to include iron-rich foods in your diet such as lean meats, poultry and fortified cereals,
            2. avoid drinking tea or coffee with meals as they can interfere with iron absorption to improve your health and prevent complications.  

            Dr. William: Do you have any drug allergies?

            Patient: No

            Dr. William: Okay then 
            1. we can try Ferrous sulfate as medication in order to treat anemia. It is an iron supplement that can help increase your hemoglobin level and improve your symptoms. 
            2. You need to take one tablet twice a day with meals for at least three months. 

            Do you have any questions or concerns, Sarah?

            Patient: No, thank you for your help, Dr. William.

            Dr. William: You are very welcome, Sarah. It was a pleasure to assist you. I hope you feel better soon. Take care and stay safe.

            End of example.
            Example:
            Conversation history:
            Dr. William: Hello, I am Dr. William, a General Physician. How are you feeling today?
            Patient: My mom took 60 pills of Valium What should i do?
            Dr. William: This is an Emergency Situation and you should take her imediately to Hospital or call the emergency service. Please do not waste any time 
            1. Wash Hands
            2. Ensure she is breathing. If not, start CPR.
            3. Try to keep her awake if possible.
            4. Do not induce vomiting unless directed by a healthcare professional.
            5. If she becomes unconscious, turn her on her side to prevent choking.
            6. Stay calm and stay with her until help arrives..
            End of example.
            Example:
            Conversation history:
            Patient: I have pain in my chest and arm what should i do ??
            Dr. William: This is an emergency situation. You may be having a heart attack. Please call 911 immediately and do not ignore these symptoms. Here's what you can do while waiting for help to arrive:
            1. Try to stay calm and take deep, slow breaths.
            2. If possible, try to sit down and rest.
            3. If you have aspirin available and are not allergic to it, chew one 325 mg tablet.
            4. Do not attempt to drive yourself to the hospital.
            6. If you become unconscious, the person with you should start doing chest compressions and rescue breaths if they know how to do CPR.
            Please stay on the line with the emergency operator and follow their instructions. They can provide you with the most appropriate advice for your situation.
            End of example.



            Current conversation stage: 
            {conversation_stage}
            Conversation history: 
            {conversation_history}
            {physician_name}:
                """
            )
            prompt = PromptTemplate(
                template=physician_agent_inception_prompt,
                input_variables=[
                    "physician_name",
                    "company_name",
                    "conversation_stage",
                    "conversation_history"
                ],
            )
            logger.info("Successfully get the prompt from template physician_agent_inception_prompt")
            return cls(prompt=prompt, llm=llm, verbose=verbose)
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
            logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")


# test the intermediate chains
verbose=True
llm = ChatOpenAI(model = "gpt-3.5-turbo-16k", temperature = 0.5)

stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

sales_conversation_utterance_chain = MedicalConversationChain.from_llm(
    llm, verbose=verbose)



product_catalog='sample_product_catalog.txt'

# Set up a knowledge base
def setup_knowledge_base(product_catalog: str = None):
    """
    We assume that the product knowledge base is simply a text file.
    """
    # load product catalog
    try:
        with open(product_catalog, "r") as f:
            product_catalog = f.read()

        text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        texts = text_splitter.split_text(product_catalog)

        llm = ChatOpenAI(temperature=0)
        embed_model = "intfloat/e5-small-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        # embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_texts(
            texts, embeddings, collection_name="product-knowledge-base"
        )

        knowledge_base = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
        )
        logger.info("Successfully run the funtion setup_knowledge_base")
        return knowledge_base
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
        logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")


def get_tools(product_catalog):
    # query to get_tools can be used to be embedded and relevant tools found
    # see here: https://langchain-langchain.vercel.app/docs/use_cases/agents/custom_agent_with_plugin_retrieval#tool-retriever

    # we only use one tool for now, but this is highly extensible!
    try:
        knowledge_base = setup_knowledge_base(product_catalog)
        tools = [
            Tool(
                name="ProductSearch",
                func=knowledge_base.run,
                description="useful for when you need to answer questions about product information",
            )
        ]
        logger.info("Successfully run the funtion get_tools")
        return tools
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
        logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")

# knowledge_base = setup_knowledge_base('sample_product_catalog.txt')
# Define a Custom Prompt Template

class CustomPromptTemplateForTools(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        try:
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Set the agent_scratchpad variable to that value
            kwargs["agent_scratchpad"] = thoughts
            ############## NEW ######################
            tools = self.tools_getter(kwargs["input"])
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            )
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
            return self.template.format(**kwargs)
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
            logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
    
# Define a custom Output Parser

class SalesConvoOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"  # change for salesperson_name
    verbose: bool = False

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            if self.verbose:
                print("TEXT")
                print(text)
                print("-------")
            if f"{self.ai_prefix}:" in text:
                return AgentFinish(
                    {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
                )
            regex = r"Action: (.*?)[\n]*Action Input: (.*)"
            match = re.search(regex, text)
            if not match:
                ## TODO - this is not entirely reliable, sometimes results in an error.
                return AgentFinish(
                    {
                        "output": "I apologize, I was unable to find the answer to your question. Is there anything else I can help with?"
                    },
                    text,
                )
                # raise OutputParserException(f"Could not parse LLM output: `{text}`")
            action = match.group(1)
            action_input = match.group(2)
            return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
            logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
    @property
    def _type(self) -> str:
        return "sales-agent"

PHYSICIAN_AGENT_TOOLS_PROMPT = """
You are Virtual Doctor helping a Psychiatrist and General Physician who specializes in providing healthcare to patients of all ages and genders. Never forget your name is Dr. {physician_name}.
You work at company named {company_name}. {company_name}'s business is providing healthcare services. Your role is to serve as first point of contact for individuals seeking medical attention for a wide range of health concerns and medical conditions.
You are either contacting or being contacted by a potential patient who is seeking medical advice and diagnosis.  

Keep your responses concise to retain the user's attention.
You must respond according to the previous conversation history and the stage of the conversation you are at. ((Don't schedule an appointment with General Physician in response. Also, don't recommend for physical checkup.))
You must take gynaecological and Obstetrics history from female patient before providing a final diagnosis.
You must provide a final diagnosis, treatment strategy response which may include off-the-shelf medications ((((((You must ask any drug allergies before  suggesting/giving any off-the-shelf medications)))))), lifestyle changes, and home remedies before ending/closing the conversation.  
Only generate one response at a time! When you are done generating, end with '<END_OF_TURN>' to give the user a chance to respond.
When the conversation is over, then end with <END_OF_CALL>
Always think about at which conversation stage you are at before answering:
"""+conversation_string+"""
9: End conversation: The physician and Psychiatrist has provided either diagnosis. The Psychiatrist and physician will end the conversation. The Psychiatrist and physician will remain on this conversation stage.

STAGES
TOOLS:
------

{physician_name} has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tools}
Action Input: the input to the action, always a simple string input
Observation: the result of the action
```

If the result of the action is "I don't know." or "Sorry I don't know", then you have to say that to the user as described in the next sentence.
When you have a response to say to the Human, or if you do not need to use a tool, or if tool did not help, you MUST use the format:

```
Thought: Do I need to use a tool? No
{physician_name}: [your response here, if previously used a tool, rephrase latest observation, if unable to find the answer, say it]
```

You must respond according to the previous conversation history and the stage of the conversation you are at.
Only generate one response at a time and act as {physician_name} only!

Begin!

Previous conversation history:
{conversation_history}

{physician_name}:
{agent_scratchpad}
"""
# print(PHYSICIAN_AGENT_TOOLS_PROMPT)
class SalesGPT(Chain, BaseModel):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    current_conversation_stage: str = '1'
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_conversation_utterance_chain: MedicalConversationChain = Field(...)

    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    use_tools: bool = False

    physician_name: str = "Ben"
    company_name: str = "InnoTech"


    def retrieve_conversation_stage(self, key):
        return conversation_stages.get(key, '1')
    
    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage= self.retrieve_conversation_stage('1')
        self.conversation_history = []

    def determine_conversation_stage(self):
        try:
            conversation_stage_id = self.stage_analyzer_chain.run(
                conversation_history='"\n"'.join(self.conversation_history), current_conversation_stage=self.current_conversation_stage)

            self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)
            print(f"Conversation Stage: {conversation_stage_id}")

            return conversation_stage_id
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
            logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
        
    def human_step(self, human_input):
        # process human input
        human_input = 'User: '+ human_input + ' <END_OF_TURN>'
        self.conversation_history.append(human_input)

    def step(self):
        ai_response = self._call(inputs={})
        return ai_response
    
    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""
        
        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.sales_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                physician_name=self.physician_name,
                company_name=self.company_name,
            )

        else:
        
            ai_message = self.sales_conversation_utterance_chain.run(
                physician_name = self.physician_name,
                company_name=self.company_name,
                conversation_history="\n".join(self.conversation_history),
                conversation_stage = self.current_conversation_stage,
            )
        
        # Add agent's response to conversation history
        # print(f'{self.physician_name}: ', ai_message.rstrip('<END_OF_TURN>'))
        agent_name = self.physician_name
        ai_message = agent_name + ": " + ai_message
        if '<END_OF_TURN>' not in ai_message:
            ai_message += ' <END_OF_TURN>'
        self.conversation_history.append(ai_message)

        return ai_message

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, verbose: bool = False, **kwargs
    ) -> "SalesGPT":
        try:
            """Initialize the SalesGPT Controller."""
            stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

            sales_conversation_utterance_chain = MedicalConversationChain.from_llm(
                    llm, verbose=verbose
                )
            
            if "use_tools" in kwargs.keys() and kwargs["use_tools"] is False:

                sales_agent_executor = None

            else:
                product_catalog = kwargs["product_catalog"]
                tools = get_tools(product_catalog)
                print("PHYSICIAN_AGENT_TOOLS_PROMPT")
                prompt = CustomPromptTemplateForTools(
                    template=PHYSICIAN_AGENT_TOOLS_PROMPT,
                    tools_getter=lambda x: tools,
                    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                    # This includes the `intermediate_steps` variable because that is needed
                    input_variables=[
                        "input",
                        "intermediate_steps",
                        "physician_name",
                        "company_name",
                        "conversation_history",
                    ],
                )
                llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

                tool_names = [tool.name for tool in tools]

                # WARNING: this output parser is NOT reliable yet
                ## It makes assumptions about output from LLM which can break and throw an error
                output_parser = SalesConvoOutputParser(ai_prefix=kwargs["physician_name"])

                sales_agent_with_tools = LLMSingleActionAgent(
                    llm_chain=llm_chain,
                    output_parser=output_parser,
                    stop=["\nObservation:"],
                    allowed_tools=tool_names,
                    verbose=verbose
                )

                sales_agent_executor = AgentExecutor.from_agent_and_tools(
                    agent=sales_agent_with_tools, tools=tools, verbose=verbose
                )


            return cls(
                stage_analyzer_chain=stage_analyzer_chain,
                sales_conversation_utterance_chain=sales_conversation_utterance_chain,
                sales_agent_executor=sales_agent_executor,
                verbose=verbose,
                **kwargs,
            )
        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
            logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
# Set up of your agent


# Agent characteristics - can be modified
config = dict(
physician_name = "William",
company_name="InnoTech",
conversation_history=[],
conversation_stage = conversation_stages.get('1', "Patient Profile: Start the conversation by introducing yourself as a General Physician. Ask the patient about the Name, Age, Gender and Occupation. Do not move to the next stage until the patient provides Age, Gender and Occupation. Ensure that the patient has provided Age, Gender and Occupation information before moving to next conversation stage"),
use_tools=False
)



def get_doctor_field_and_department(history):
    try:
        prompt = PromptTemplate(input_variables=["history"],template="""You are an AI assisstant that just give the Doctor Department and Doctor field from below list based on the conversation between Ben and User that is {history}. Remember your response strictly just give the 1 Doctor Department and 1 Doctor field it's cumpolsary to add comma(,) in between them for example: Cardiologist Department, Cardiologist.
        Doctor Department:
            Medical/Surgical Units, Intensive Care Units (ICUs), Labor and Delivery Unit, Pediatrics Department, Obstetrics and Gynecology Department (OB/GYN), Radiology Department, Pathology Department, Pharmacy Department, Operating Room (OR), Anesthesia Department, Cardiology Department, Oncology Department, Neurology Department, Orthopedics Department, Physical Therapy and Rehabilitation Department, Nutrition and Dietetics Department, Infection Control Department, Pain Management Department, Breast Health Clinic, Rheumatology Department, Gynecology Department, Gastroenterology Department, General Surgery Department, Otolaryngology (ENT) Department, General Medicine Department, Internal Medicine Department, Sleep Medicine Department, Psychiatry Department, Psychology Department, Occupational Health Department, Infectious Diseases Department, Pulmonology Department, Urology Department, Dermatology Department, ENT (Ear, Nose, and Throat) Department
        Doctor field: 
            Surgeon, Intensivist, Obstetrician, Pediatrician, OB/GYN Specialist, Radiologist, Pathologist, Pharmacist, Surgeon, Anesthesiologist, Cardiologist, Oncologist, Neurologist, Orthopedic Surgeon, Physical Therapist, Dietitian, Infection Control Specialist, Pain Management Specialist, Breast Health Specialist, Rheumatologist, Gynecologist, Gastroenterologist, General Surgeon, ENT Specialist, General Practitioner (GP), Internist, Sleep Specialist, Psychiatrist, Psychologist, Occupational Health Specialist, Infectious Disease Specialist, Pulmonologist, Urologist, Dermatologist.
    """)
        # prompt.format(history=history)
        # llm = OpenAI(model = "gpt-3.5-turbo-16k")
        llm = ChatOpenAI(model = "gpt-4",temperature=0.1)
        chain=LLMChain(llm=llm,prompt=prompt)
        res=chain.run(history)
        logger.info("Successfully get the response from funtion get_doctor_field_and_department")
        print("*******************************")
        print("*******************************")
        print(res)
        print("*******************************")
        return res.replace("\n","").split(",")
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
        logger.critical(f"Error: {str(e)} in file_name: {str(fname)} line no: {str(exc_tb.tb_lineno)}")
