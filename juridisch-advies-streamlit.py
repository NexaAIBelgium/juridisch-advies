#!/usr/bin/env python3
"""
Juridisch Multi-Agent Adviessysteem - Streamlit Implementatie
Een interactieve webapplicatie voor juridische analyse met 4 gespecialiseerde AI-agenten
"""

import streamlit as st
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
from pathlib import Path
import google.generativeai as genai
from dataclasses import dataclass, asdict
import time
from PIL import Image
import io

# ================================
# OCR FUNCTIONALITEIT
# ================================

def extract_text_from_image(image_file, describe_images=True) -> str:
    """Extract text from image using Gemini's vision capabilities"""
    try:
        # Convert uploaded file to PIL Image
        image = Image.open(image_file)
        
        # Configure model for vision
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for OCR and image description
        if describe_images:
            prompt = """Analyseer deze afbeelding en doe het volgende:

1. TEKSTEXTRACTIE: Extract ALLE tekst die je in de afbeelding ziet. 
   - Behoud de originele structuur
   - Include alle details, data, namen, bedragen, en juridische referenties
   - Tekst in het Nederlands/Vlaams

2. AFBEELDINGSBESCHRIJVING: Als er visuele elementen zijn (foto's, diagrammen, schade, objecten, personen, locaties):
   - Beschrijf wat je ziet in detail
   - Voor schadeclaims: beschrijf de aard en omvang van de schade
   - Voor ongelukken: beschrijf de situatie, posities, omstandigheden
   - Voor producten: beschrijf de staat, gebreken, of problemen
   - Wees objectief en feitelijk in je beschrijvingen

Format je antwoord als volgt:
=== GEËXTRAHEERDE TEKST ===
[alle tekst uit het document]

=== VISUELE ELEMENTEN ===
[beschrijving van niet-tekstuele elementen, indien aanwezig]
"""
        else:
            prompt = """Please extract ALL text from this image. 
        Format the text maintaining the original structure as much as possible.
        If there are multiple sections, separate them clearly.
        Include all details, dates, names, amounts, and legal references.
        
        Important: Extract text in the original language (likely Dutch/Flemish).
        """
        
        # Generate response
        response = model.generate_content([prompt, image])
        
        if response.text:
            return response.text
        else:
            return "Geen tekst of visuele elementen gevonden in de afbeelding."
            
    except Exception as e:
        st.error(f"Fout bij verwerking: {str(e)}")
        return ""

def process_multiple_documents(uploaded_files, describe_images=True) -> str:
    """Process multiple document images and combine the extracted text"""
    all_text = []
    
    progress_bar = st.progress(0)
    
    for idx, file in enumerate(uploaded_files):
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
        with st.spinner(f"Verwerken van {file.name}..."):
            extracted_content = extract_text_from_image(file, describe_images)
            
            if extracted_content:
                all_text.append(f"\n{'='*50}\nDocument: {file.name}\n{'='*50}\n{extracted_content}\n")
    
    progress_bar.empty()
    
    return "\n".join(all_text)

# ================================
# PAGINA CONFIGURATIE
# ================================

st.set_page_config(
    page_title="Juridisch AI Advies",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CONFIGURATIE EN SETUP
# ================================

# Initialize session state
if 'analyses_completed' not in st.session_state:
    st.session_state.analyses_completed = False
if 'agent_outputs' not in st.session_state:
    st.session_state.agent_outputs = {}
if 'final_advice' not in st.session_state:
    st.session_state.final_advice = ""

# Configureer Gemini
def configure_gemini():
    """Configureer Gemini API"""
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        st.error("⚠️ Geen Google API key gevonden! Stel deze in via Streamlit secrets of environment variables.")
        st.stop()
    
    genai.configure(api_key=api_key)
    return True

# Model configuratie
LITE_MODEL = "gemini-2.0-flash-lite"  # Voor alle analyse agents
ADVANCED_MODEL = "gemini-2.0-flash-lite"  # Voor synthesizer en advies generator

# ================================
# DATA CLASSES
# ================================

@dataclass
class CasusInput:
    """Structuur voor de advocaat input"""
    client_naam: str
    client_rol: str
    tegenpartij_naam: str
    tegenpartij_rol: str
    situatie_samenvatting: str
    doel_client: str
    vorderingen: List[str]
    feitenrelaas: str
    bewijsstukken: List[str]

@dataclass
class AnalyseResultaat:
    """Structuur voor het finale resultaat"""
    transcript_analyse: str
    concept_adviesnota: str
    timestamp: str
    dossier_info: Dict[str, str]

# ================================
# HELPER FUNCTIES
# ================================

def call_gemini(system_prompt: str, user_prompt: str, model_name: str = LITE_MODEL) -> str:
    """Roep Gemini model aan met gegeven prompts"""
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
            safety_settings=safety_settings
        )
        
        full_prompt = f"""System: {system_prompt}

User: {user_prompt}"""
        
        response = model.generate_content(full_prompt)
        
        if not response.parts:
            return "ERROR: Geen response ontvangen"
        
        return response.text
        
    except Exception as e:
        return f"ERROR: {str(e)}"

# ================================
# AGENT IMPLEMENTATIES
# ================================

class Agent1_AnalytischeJurist:
    """Agent 1: Bouwt het pleidooi pro-cliënt"""
    
    def __init__(self):
        self.name = "Agent 1 - Analytische Jurist"
        self.model = LITE_MODEL
        self.system_prompt = """Je bent een ervaren Belgische advocaat die de sterkst mogelijke casus voor de cliënt opbouwt. 
Je structureert je analyse volgens 5 pijlers: Narratief, Juridische Interpretatie, Toerekening aan Tegenpartij, 
Aanval op Vordering, en Bewijskracht. Je bent grondig maar beknopt."""
    
    def analyze(self, casus: CasusInput) -> str:
        """Analyseer de casus vanuit pro-cliënt perspectief"""
        
        user_prompt = f"""Analyseer deze casus en bouw het sterkste pleidooi voor onze cliënt:

CLIËNT: {casus.client_naam} ({casus.client_rol})
TEGENPARTIJ: {casus.tegenpartij_naam} ({casus.tegenpartij_rol})

SITUATIE: {casus.situatie_samenvatting}

DOEL CLIËNT: {casus.doel_client}

VORDERINGEN TEGENPARTIJ: {', '.join(casus.vorderingen)}

FEITEN: {casus.feitenrelaas}

BEWIJSSTUKKEN: {', '.join(casus.bewijsstukken)}

Structureer je analyse volgens deze 5 pijlers:

1. NARRATIEF & FACTUAL FRAMING
- Presenteer het feitenrelaas vanuit het perspectief van de cliënt
- Contextualiseer ongunstige feiten

2. GUNSTIGE JURIDISCHE INTERPRETATIE
- Selecteer relevante wetsartikelen en rechtspraak
- Interpreteer deze maximaal gunstig voor de cliënt

3. TOEREKENING AAN TEGENPARTIJ
- Identificeer fouten/nalatigheden van de tegenpartij
- Analyseer hun eigen verantwoordelijkheid

4. AANVAL OP DE VORDERING
- Betwist systematisch: fout, schade, causaal verband
- Argumenteer waarom vorderingen ongegrond zijn

5. BEWIJSKRACHTIGE ARGUMENTATIE
- Verwijs naar ondersteunende bewijsstukken
- Wijs op bewijslacunes bij tegenpartij

Wees concreet en verwijs naar Belgisch recht."""
        
        return call_gemini(self.system_prompt, user_prompt, self.model)

class Agent2_AdvocaatDuivel:
    """Agent 2: Identificeert zwaktes en risico's"""
    
    def __init__(self):
        self.name = "Agent 2 - Advocaat van de Duivel"
        self.model = LITE_MODEL
        self.system_prompt = """Je bent de beste advocaat van de tegenpartij. Je identificeert genadeloos alle zwaktes 
in de positie van onze cliënt en alle risico's. Je denkt als een tegenstander die wil winnen."""
    
    def analyze(self, casus: CasusInput, agent1_output: str) -> str:
        """Analyseer zwaktes en risico's"""
        
        user_prompt = f"""Analyseer deze casus TEGEN onze cliënt. Je hebt ook de analyse van Agent 1 gezien:

CASUS INFO:
Cliënt: {casus.client_naam} ({casus.client_rol})
Tegenpartij: {casus.tegenpartij_naam} ({casus.tegenpartij_rol})
Situatie: {casus.situatie_samenvatting}
Vorderingen: {', '.join(casus.vorderingen)}

AGENT 1 ANALYSE (pro-cliënt):
{agent1_output[:1000]}...

Structureer je tegenanalyse volgens deze 5 pijlers:

1. STERKST MOGELIJKE TEGENARGUMENT
- Formuleer het meest overtuigende argument voor de tegenpartij
- Gebruik de gunstigste interpretatie van feiten en recht voor hen

2. IDENTIFICATIE VAN ONZE ZWAKTES
- Welke feiten zijn objectief ongunstig?
- Welke acties zijn moeilijk te verdedigen?
- Welke juridische argumenten zijn wankel?

3. ANALYSE VAN BEWIJSRISICO'S
- Welk bewijs missen we?
- Welke bewijsstukken van tegenpartij zijn schadelijk?
- Kunnen we aan de bewijslast voldoen?

4. WORST-CASE SCENARIO
- Maximale financiële blootstelling?
- Niet-financiële risico's?
- Precedentwerking?

5. ANTICIPATIE OP ONS VERWEER
- Hoe zal tegenpartij onze argumenten weerleggen?
- Welke tegenargumenten zijn het sterkst?

Wees meedogenloos kritisch."""
        
        return call_gemini(self.system_prompt, user_prompt, self.model)

class Agent3_ProcedureleStrateeg:
    """Agent 3: Zoekt procedurele knock-out argumenten"""
    
    def __init__(self):
        self.name = "Agent 3 - Procedurele Strateeg"
        self.model = LITE_MODEL
        self.system_prompt = """Je bent een expert in Belgisch procesrecht. Je focust uitsluitend op procedurele, 
formele en contractuele argumenten die de vordering kunnen doen falen zonder inhoudelijke behandeling."""
    
    def analyze(self, casus: CasusInput) -> str:
        """Analyseer procedurele aspecten"""
        
        user_prompt = f"""Analyseer procedurele knock-out mogelijkheden voor deze casus:

PARTIJEN: {casus.client_naam} vs {casus.tegenpartij_naam}
VORDERINGEN: {', '.join(casus.vorderingen)}
FEITEN: {casus.feitenrelaas}
BEWIJSSTUKKEN: {', '.join(casus.bewijsstukken)}

Analyseer systematisch:

1. VERJARING & VERVALTERMIJNEN
- Identificeer toepasselijke termijnen (wettelijk/contractueel)
- Bepaal startdata (gunstig vs conservatief)
- Analyseer stuiting/schorsing

2. KLACHTPLICHT & MELDINGSTERMIJNEN
- Is er een klachtplicht?
- Tijdig geklaagd?
- Correcte wijze en inhoud?

3. CONTRACTUELE ANALYSE
- Exoneratiebedingen
- Garantieclausules
- Boetebedingen
- Forum-/rechtskeuzebedingen
- Finale kwijting
- Pre-processuele vereisten

4. FORMELE VEREISTEN
- Ingebrekestelling correct?
- Alle voorwaarden vervuld?

5. BEVOEGDHEID & ONTVANKELIJKHEID
- Materiële/territoriale bevoegdheid
- Belang en hoedanigheid

Voor elk punt: geef potentie aan (HOOG/GEMIDDELD/LAAG) met juridische basis."""
        
        return call_gemini(self.system_prompt, user_prompt, self.model)

class Agent4_Synthesizer:
    """Agent 4: Weegt alle analyses en formuleert strategie"""
    
    def __init__(self):
        self.name = "Agent 4 - Eindverantwoordelijke Adviseur"
        self.model = ADVANCED_MODEL
        self.system_prompt = """Je bent de senior partner die alle analyses integreert tot een coherent strategisch advies. 
Je weegt de pro's (Agent 1), contra's (Agent 2) en procedurele aspecten (Agent 3) om tot een 
evenwichtige kansinschatting en optimale strategie te komen."""
    
    def synthesize(self, casus: CasusInput, agent1: str, agent2: str, agent3: str) -> str:
        """Creëer synthese van alle analyses"""
        
        user_prompt = f"""Integreer de analyses van alle agenten tot een coherente strategie:

CASUS: {casus.client_naam} vs {casus.tegenpartij_naam}
DOEL: {casus.doel_client}

AGENT 1 (Pro-cliënt):
{agent1[:1500]}...

AGENT 2 (Risico's):
{agent2[:1500]}...

AGENT 3 (Procedureel):
{agent3[:1500]}...

SYNTHESISEER:

1. WEGING VAN ARGUMENTEN
- Welke argumenten van Agent 1 zijn het sterkst?
- Welke risico's van Agent 2 zijn reëel?
- Welke procedurele punten van Agent 3 zijn kansrijk?

2. KANSINSCHATTING
- Geef een percentage of kwalitatieve inschatting
- Motiveer deze inschatting
- Identificeer het hoofdrisico

3. STRATEGISCHE AANBEVELING
- Formuleer de optimale strategie
- Prioriteer de stappen
- Geef concrete acties

Wees evenwichtig en realistisch in je beoordeling."""
        
        return call_gemini(self.system_prompt, user_prompt, self.model)

# ================================
# ADVIES GENERATOR
# ================================

class AdviesGenerator:
    """Genereert het finale juridische advies volgens template"""
    
    def __init__(self):
        self.model = ADVANCED_MODEL
    
    def genereer_uitgebreid_advies(self, casus: CasusInput, all_analyses: Dict[str, str], timestamp: str) -> str:
        """Genereer uitgebreide adviesnota exact volgens de template"""
        
        datum = datetime.now().strftime("%d/%m/%Y")
        
        system_prompt = """Je bent een senior juridisch adviseur die een professionele adviesnota opstelt voor een Belgische juridische context.
Je moet EXACT de gegeven template volgen, inclusief alle asterisken (*) voor bullet points en formatting.
Integreer de analyses van alle agents in de juiste secties volgens de template instructies.
Wees uitgebreid en concreet in je uitwerking."""
        
        user_prompt = f"""Genereer een juridisch advies EXACT volgens deze template. Gebruik de analyses van de agents om de template in te vullen:

CASUS INFO:
- Cliënt: {casus.client_naam} ({casus.client_rol})
- Tegenpartij: {casus.tegenpartij_naam} ({casus.tegenpartij_rol})
- Situatie: {casus.situatie_samenvatting}
- Doel: {casus.doel_client}
- Vorderingen: {', '.join(casus.vorderingen)}

AGENT ANALYSES (gebruik deze info om de template in te vullen):
Agent 1 (Pro-cliënt argumenten): {all_analyses.get('sterke_punten', '')[:1500]}
Agent 2 (Risico's): {all_analyses.get('risicos', '')[:1500]}
Agent 3 (Procedurele kansen): {all_analyses.get('procedurele_kansen', '')[:1500]}
Agent 4 (Synthese): {all_analyses.get('synthese', '')[:1500]}

VEREISTE OUTPUT (volg EXACT deze template):

**DEEL B: CONCEPT ADVIESNOTA / STRATEGISCH MEMO**
==================================================

**AAN:** Behandelend Advocaat
**VAN:** AI Sparringpartner
**DOSSIER:** {casus.client_naam}
**DATUM:** {datum}
**BETREFT:** Analyse rechtspositie en strategische opties inzake {casus.situatie_samenvatting[:80]}...

### 1. KERN VAN DE ZAAK EN ADVIESVRAAG
*   **Conflict:** [Zeer korte samenvatting van het geschil tussen {casus.client_naam} en {casus.tegenpartij_naam}]
*   **Adviesvraag:** Wat is de juridische sterkte van onze positie en welke strategische stappen zijn aan te bevelen om het doel van de cliënt te bereiken?

### 2. SAMENVATTING RELEVANTE FEITEN
[Presenteer een beknopte, objectieve en chronologische samenvatting van de juridisch relevante feiten uit de casus]

### 3. JURIDISCHE ANALYSE

**3.1. Toepasselijk Kader**
*   **Rechtsregels:** [Lijst relevante Belgische wetsartikelen zoals art. 1641 oud BW voor verborgen gebreken, art. 1382 oud BW voor aansprakelijkheid, etc.]
*   **Relevante Jurisprudentie:** [Vermeld 1-2 relevante Cassatie-arresten met datum en nummer]

**3.2. Analyse Sterktes en Zwaktes (Debat samengevat)**
*   **Procedurele Kansen (Knock-outs):**
    *   [Gebruik info van Agent 3 - formuleer belangrijkste procedurele verweren]
*   **Inhoudelijke Argumenten pro Cliënt (Onze Case):**
    *   [Gebruik info van Agent 1 - eerste sterke punt]
    *   [Gebruik info van Agent 1 - tweede sterke punt]
*   **Inhoudelijke Tegenargumenten & Risico's (Case van de Tegenpartij):**
    *   [Gebruik info van Agent 2 - sterkste tegenargument]
    *   [Gebruik info van Agent 2 - belangrijkste risico]

**3.3. Analyse Bewijspositie Tegenpartij (Gaten en Kansen)**
*   **Bewijslast Tegenpartij:** De tegenpartij draagt de bewijslast voor [specificeer exact welke stellingen zij moeten bewijzen]
*   **Geïdentificeerde Bewijsgaten:** Op basis van de nu beschikbare stukken, ontbreekt er bewijs voor de volgende cruciale stellingen van de tegenpartij:
    *   Stelling 1: "[Specifieke bewering]". **Ontbrekend bewijs:** [Wat ontbreekt]
    *   Stelling 2: "[Andere bewering]". **Ontbrekend bewijs:** [Wat ontbreekt]
*   **Strategisch Informatieverzoek:** Het is aan te bevelen de tegenpartij te verzoeken de volgende stukken over te leggen:
    1.  [Specifiek document dat hun stelling moet ondersteunen]
    2.  [Ander relevant document]

### 4. GEWOGEN CONCLUSIE EN KANSINSCHATTING
*   **Synthese:** [Geef gewogen oordeel op basis van Agent 4 synthese, integreer procedurele en inhoudelijke aspecten]
*   **Kansinschatting (indicatief):**
    *   Succes in procedure: [Percentage of kwalitatieve inschatting met korte motivering]
    *   Belangrijkste risico: [Identificeer het hoofdrisico voor onze positie]

### 5. STRATEGISCHE OPTIES EN AANBEVELING
**Optie 1: [Geef concrete naam, bv. "Procedureel verweer op basis van klachtplicht"] (Aanbevolen)**
*   **Beschrijving:** [Beschrijf de strategie concreet]
*   **Concrete Stappen:** 
    1.  [Eerste concrete stap]
    2.  [Tweede concrete stap]
    3.  [Derde concrete stap]
*   **Voordelen:** [Lijst 2-3 voordelen]

**Optie 2: [Andere strategie, bv. "Inhoudelijk verweer met openheid voor minnelijke regeling"]**
*   **Beschrijving:** [Beschrijf deze alternatieve strategie]
*   **Concrete Stappen:**
    1.  [Eerste stap]
    2.  [Tweede stap]
*   **Voordelen:** [Lijst voordelen]

**Aanbeveling:**
[Motiveer waarom Optie 1 wordt aanbevolen, verwijs naar de sterktes uit de analyse]

BELANGRIJK: Volg deze template EXACT, inclusief de bullet points met asterisken (*), de nummering, en de structuur."""
        
        return call_gemini(system_prompt, user_prompt, self.model)

# ================================
# STREAMLIT UI COMPONENTEN
# ================================

def display_header():
    """Toon de header van de applicatie"""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# ⚖️")
    with col2:
        st.title("Juridisch Multi-Agent Adviessysteem")
        st.markdown("*AI-powered juridische analyse met 4 gespecialiseerde agenten*")
    
    st.divider()

def display_sidebar():
    """Configureer en toon de sidebar"""
    with st.sidebar:
        st.header("📋 Systeem Informatie")
        
        st.info("""
        **Dit systeem bevat 4 AI-agenten:**
        
        🔵 **Agent 1**: Analytische Jurist
        - Bouwt sterkste pleidooi pro-cliënt
        
        🔴 **Agent 2**: Advocaat van de Duivel  
        - Identificeert zwaktes en risico's
        
        🟡 **Agent 3**: Procedurele Strateeg
        - Zoekt procedurele knock-out kansen
        
        🟢 **Agent 4**: Senior Adviseur
        - Integreert analyses tot strategie
        
        ---
        
        **📷 OCR Functionaliteit:**
        - Upload foto's van documenten
        - AI extraheert automatisch tekst
        - Intelligent formulier invullen
        """)
        
        st.divider()
        
        st.header("⚙️ Instellingen")
        
        # Voorbeeld casus optie
        if st.button("📝 Laad Voorbeeld Casus"):
            st.session_state.load_example = True
            
        # Reset knop
        if st.button("🔄 Reset Alles"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

def get_casus_input():
    """Verzamel casus informatie via formulier"""
    st.header("📝 Casus Informatie")
    
    # Document upload sectie
    with st.expander("📷 Upload Documenten (OCR + Beeldbeschrijving)", expanded=False):
        st.info("""
        Upload foto's of scans van juridische documenten. De AI zal:
        - 📄 Tekst automatisch extraheren
        - 🖼️ Afbeeldingen beschrijven (schade, locaties, producten, etc.)
        - 🔍 Beide integreren voor complete documentanalyse
        """)
        
        uploaded_files = st.file_uploader(
            "Selecteer afbeeldingen",
            type=['png', 'jpg', 'jpeg', 'webp'],
            accept_multiple_files=True,
            help="Upload contracten, brieven, facturen, schadebeelden, foto's van producten/locaties, etc."
        )
        
        if uploaded_files:
            # Opties voor verwerking
            col1, col2 = st.columns(2)
            with col1:
                describe_images = st.checkbox(
                    "🖼️ Beschrijf ook visuele elementen",
                    value=True,
                    help="Laat AI foto's en afbeeldingen beschrijven (nuttig voor schadebeelden, ongeluksfoto's, etc.)"
                )
            
            with col2:
                if st.button("🔍 Start Document Verwerking", type="primary"):
                    extracted_content = process_multiple_documents(uploaded_files, describe_images)
                    st.session_state.ocr_text = extracted_content
                    
                    # Toon extracted content
                    st.success(f"✅ Verwerking compleet voor {len(uploaded_files)} document(en)")
                    
                    # Preview van documenten
                    with st.expander("👁️ Preview Documenten", expanded=False):
                        cols = st.columns(min(len(uploaded_files), 3))
                        for idx, (col, file) in enumerate(zip(cols, uploaded_files[:3])):
                            with col:
                                image = Image.open(file)
                                st.image(image, caption=file.name, use_column_width=True)
                        if len(uploaded_files) > 3:
                            st.info(f"... en {len(uploaded_files) - 3} meer documenten")
                    
                    # Toon geëxtraheerde content
                    with st.expander("📝 Bekijk Geëxtraheerde Inhoud", expanded=True):
                        # Tabs voor verschillende views
                        tab1, tab2 = st.tabs(["📄 Volledige Output", "🔍 Per Document"])
                        
                        with tab1:
                            st.text_area("Complete inhoud", extracted_content, height=400)
                        
                        with tab2:
                            # Split content per document
                            documents = extracted_content.split("\n" + "="*50 + "\n")
                            for doc in documents[1:]:  # Skip first empty split
                                if doc.strip():
                                    doc_name = doc.split('\n')[0].replace("Document: ", "")
                                    with st.expander(f"📄 {doc_name}"):
                                        st.text(doc)
                    
                    # Analyse tips
                    st.info("""
                    💡 **Tips voor juridische documentanalyse:**
                    - Bij **schadebeelden**: De AI beschrijft de aard en omvang van de schade
                    - Bij **contracten**: Alle clausules en bedingen worden geëxtraheerd  
                    - Bij **correspondentie**: Data en inhoud worden gestructureerd weergegeven
                    - Bij **bewijsmateriaal**: Visuele elementen worden objectief beschreven
                    """)
    
    # Check voor voorbeeld casus
    if st.session_state.get('load_example', False):
        example_data = {
            'client_naam': 'NV TechStart',
            'client_rol': 'Opdrachtnemer/Dienstverlener',
            'tegenpartij_naam': 'NV GlobalCorp',
            'tegenpartij_rol': 'Opdrachtgever',
            'situatie_samenvatting': 'GlobalCorp beëindigde eenzijdig een IT-ontwikkelingscontract met TechStart wegens beweerde wanprestatie. TechStart stelt dat de vertraging te wijten was aan gebrekkige specificaties van GlobalCorp. Er staat €150.000 aan facturen open.',
            'doel_client': 'Betaling verkrijgen van openstaande facturen en schadevergoeding voor contractbreuk',
            'vorderingen': ['Terugbetaling voorschotten: €50.000', 'Schadevergoeding wegens wanprestatie: €200.000', 'Contractuele boete: €25.000'],
            'feitenrelaas': '''Chronologie:
- 01/01/2024: Ondertekening contract voor ontwikkeling ERP-systeem
- 15/03/2024: Eerste milestone opgeleverd conform planning
- 01/05/2024: GlobalCorp wijzigt fundamenteel de specificaties
- 15/06/2024: TechStart meldt vertraging door scopewijziging
- 01/07/2024: GlobalCorp beëindigt contract wegens vertraging
- 15/07/2024: Ingebrekestelling door GlobalCorp''',
            'bewijsstukken': ['Ondertekend contract dd. 01/01/2024', 'Email correspondentie over scopewijzigingen', 'Projectdocumentatie en deliverables', 'Facturen voor verrichte werkzaamheden', 'Brief contractbeëindiging dd. 01/07/2024']
        }
        st.session_state.load_example = False
    else:
        example_data = {}
    
    # AI-assisted form filling
    if st.session_state.get('ocr_text'):
        with st.expander("🤖 AI-Assistent voor Formulier Invullen", expanded=False):
            if st.button("🪄 Analyseer Documenten en Vul Formulier"):
                with st.spinner("AI analyseert de documenten..."):
                    # Use Gemini to analyze the OCR text and extract relevant information
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    analysis_prompt = f"""Analyseer de volgende geëxtraheerde tekst uit juridische documenten en identificeer:
                    
1. Namen van partijen (client en tegenpartij)
2. Rollen van partijen (koper/verkoper, huurder/verhuurder, etc.)
3. Samenvatting van het conflict
4. Vorderingen met bedragen
5. Chronologie van gebeurtenissen
6. Genoemde bewijsstukken

Geëxtraheerde tekst:
{st.session_state.ocr_text}

Geef het resultaat in JSON formaat:
{{
    "client_naam": "",
    "client_rol": "",
    "tegenpartij_naam": "",
    "tegenpartij_rol": "",
    "situatie_samenvatting": "",
    "vorderingen": [],
    "feiten_chronologie": "",
    "bewijsstukken": []
}}"""
                    
                    try:
                        response = model.generate_content(analysis_prompt)
                        # Parse the JSON response
                        import re
                        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                        if json_match:
                            extracted_data = json.loads(json_match.group())
                            st.session_state.ai_extracted_data = extracted_data
                            st.success("✅ AI heeft de documenten geanalyseerd!")
                            st.json(extracted_data)
                            st.info("De geëxtraheerde informatie is opgeslagen. Je kunt het formulier nu invullen met deze gegevens.")
                    except Exception as e:
                        st.error(f"Fout bij AI analyse: {str(e)}")
    
    # Get AI extracted data if available
    ai_data = st.session_state.get('ai_extracted_data', {})
    
    with st.form("casus_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛡️ Uw Cliënt")
            client_naam = st.text_input(
                "Naam cliënt", 
                value=example_data.get('client_naam', ai_data.get('client_naam', ''))
            )
            client_rol = st.text_input(
                "Rol cliënt (bv. koper, verhuurder)", 
                value=example_data.get('client_rol', ai_data.get('client_rol', ''))
            )
            
        with col2:
            st.subheader("⚔️ De Tegenpartij")
            tegenpartij_naam = st.text_input(
                "Naam tegenpartij", 
                value=example_data.get('tegenpartij_naam', ai_data.get('tegenpartij_naam', ''))
            )
            tegenpartij_rol = st.text_input(
                "Rol tegenpartij", 
                value=example_data.get('tegenpartij_rol', ai_data.get('tegenpartij_rol', ''))
            )
        
        st.subheader("📄 Casus Details")
        
        situatie_samenvatting = st.text_area(
            "Samenvatting van de situatie",
            height=100,
            value=example_data.get('situatie_samenvatting', ai_data.get('situatie_samenvatting', '')),
            help="Geef een beknopte omschrijving van het conflict"
        )
        
        doel_client = st.text_area(
            "Doel van de cliënt",
            height=80,
            value=example_data.get('doel_client', ''),
            help="Wat wil uw cliënt bereiken?"
        )
        
        # Vorderingen
        st.subheader("💰 Vorderingen van de Tegenpartij")
        default_vorderingen = example_data.get('vorderingen', [])
        if not default_vorderingen and ai_data.get('vorderingen'):
            default_vorderingen = ai_data.get('vorderingen', [])
        
        vorderingen_text = st.text_area(
            "Vorderingen (één per regel)",
            height=100,
            value='\n'.join(default_vorderingen),
            help="Lijst alle vorderingen van de tegenpartij"
        )
        
        # Feiten
        feitenrelaas = st.text_area(
            "Feitenrelaas",
            height=200,
            value=example_data.get('feitenrelaas', ai_data.get('feiten_chronologie', '')),
            help="Chronologisch overzicht van de relevante feiten"
        )
        
        # Bewijsstukken
        default_bewijsstukken = example_data.get('bewijsstukken', [])
        if not default_bewijsstukken and ai_data.get('bewijsstukken'):
            default_bewijsstukken = ai_data.get('bewijsstukken', [])
            
        bewijsstukken_text = st.text_area(
            "Bewijsstukken (één per regel)",
            height=100,
            value='\n'.join(default_bewijsstukken),
            help="Lijst alle beschikbare bewijsstukken"
        )
        
        # Include OCR text if available
        include_ocr = st.checkbox(
            "📎 Voeg geëxtraheerde documenttekst toe aan analyse",
            value=bool(st.session_state.get('ocr_text')),
            help="De volledige geëxtraheerde tekst wordt meegestuurd naar de AI agents"
        )
        
        submitted = st.form_submit_button("🚀 Start Analyse", type="primary", use_container_width=True)
        
        if submitted:
            # Validatie
            if not all([client_naam, tegenpartij_naam, situatie_samenvatting]):
                st.error("Vul minimaal de namen van beide partijen en een situatiebeschrijving in.")
                return None
            
            # Parse vorderingen en bewijsstukken
            vorderingen = [v.strip() for v in vorderingen_text.split('\n') if v.strip()]
            bewijsstukken = [b.strip() for b in bewijsstukken_text.split('\n') if b.strip()]
            
            # Add OCR text to feitenrelaas if requested
            if include_ocr and st.session_state.get('ocr_text'):
                feitenrelaas += f"\n\n=== GEËXTRAHEERDE DOCUMENTEN ===\n{st.session_state.get('ocr_text')}"
            
            return CasusInput(
                client_naam=client_naam,
                client_rol=client_rol,
                tegenpartij_naam=tegenpartij_naam,
                tegenpartij_rol=tegenpartij_rol,
                situatie_samenvatting=situatie_samenvatting,
                doel_client=doel_client,
                vorderingen=vorderingen or ["Geen vorderingen gespecificeerd"],
                feitenrelaas=feitenrelaas,
                bewijsstukken=bewijsstukken or ["Geen bewijsstukken gespecificeerd"]
            )
    
    return None

def run_analysis(casus: CasusInput):
    """Voer de multi-agent analyse uit met progress tracking"""
    st.header("🔄 Analyse in Uitvoering")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Container voor agent outputs
    agent_container = st.container()
    
    try:
        # Agent 1
        with agent_container.expander("🔵 Agent 1 - Analytische Jurist", expanded=True):
            status_text.text("Agent 1 analyseert pro-cliënt argumenten...")
            progress_bar.progress(10)
            
            agent1 = Agent1_AnalytischeJurist()
            start_time = time.time()
            agent1_output = agent1.analyze(casus)
            elapsed = time.time() - start_time
            
            st.success(f"✅ Analyse compleet in {elapsed:.1f} seconden")
            with st.container():
                st.markdown(agent1_output)
            st.session_state.agent_outputs['agent1'] = agent1_output
        
        # Agent 2
        with agent_container.expander("🔴 Agent 2 - Advocaat van de Duivel", expanded=True):
            status_text.text("Agent 2 identificeert risico's en zwaktes...")
            progress_bar.progress(35)
            
            agent2 = Agent2_AdvocaatDuivel()
            start_time = time.time()
            agent2_output = agent2.analyze(casus, agent1_output)
            elapsed = time.time() - start_time
            
            st.success(f"✅ Analyse compleet in {elapsed:.1f} seconden")
            with st.container():
                st.markdown(agent2_output)
            st.session_state.agent_outputs['agent2'] = agent2_output
        
        # Agent 3
        with agent_container.expander("🟡 Agent 3 - Procedurele Strateeg", expanded=True):
            status_text.text("Agent 3 zoekt procedurele knock-out mogelijkheden...")
            progress_bar.progress(60)
            
            agent3 = Agent3_ProcedureleStrateeg()
            start_time = time.time()
            agent3_output = agent3.analyze(casus)
            elapsed = time.time() - start_time
            
            st.success(f"✅ Analyse compleet in {elapsed:.1f} seconden")
            with st.container():
                st.markdown(agent3_output)
            st.session_state.agent_outputs['agent3'] = agent3_output
        
        # Agent 4
        with agent_container.expander("🟢 Agent 4 - Senior Adviseur", expanded=True):
            status_text.text("Agent 4 integreert alle analyses...")
            progress_bar.progress(85)
            
            agent4 = Agent4_Synthesizer()
            start_time = time.time()
            agent4_output = agent4.synthesize(casus, agent1_output, agent2_output, agent3_output)
            elapsed = time.time() - start_time
            
            st.success(f"✅ Synthese compleet in {elapsed:.1f} seconden")
            with st.container():
                st.markdown(agent4_output)
            st.session_state.agent_outputs['agent4'] = agent4_output
        
        # Genereer finale advies
        status_text.text("Genereren van uitgebreid juridisch advies...")
        progress_bar.progress(95)
        
        generator = AdviesGenerator()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        synthese_data = {
            'synthese': agent4_output,
            'procedurele_kansen': agent3_output,
            'sterke_punten': agent1_output,
            'risicos': agent2_output
        }
        
        final_advice = generator.genereer_uitgebreid_advies(casus, synthese_data, timestamp)
        st.session_state.final_advice = final_advice
        st.session_state.casus_info = asdict(casus)
        st.session_state.timestamp = timestamp
        
        progress_bar.progress(100)
        status_text.text("✅ Analyse compleet!")
        
        st.session_state.analyses_completed = True
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Fout tijdens analyse: {str(e)}")
        st.stop()

def display_results():
    """Toon de resultaten van de analyse"""
    st.header("📊 Analyse Resultaten")
    
    # Tabs voor verschillende views
    tab1, tab2, tab3 = st.tabs(["📄 Juridisch Advies", "🔍 Agent Analyses", "💾 Download"])
    
    with tab1:
        st.markdown("### Uitgebreid Juridisch Advies")
        
        # Advies in een mooi gestylede container
        with st.container():
            st.markdown(st.session_state.final_advice)
    
    with tab2:
        st.markdown("### Individuele Agent Analyses")
        
        # Toon elke agent output
        agents = [
            ("🔵 Agent 1 - Analytische Jurist", 'agent1'),
            ("🔴 Agent 2 - Advocaat van de Duivel", 'agent2'),
            ("🟡 Agent 3 - Procedurele Strateeg", 'agent3'),
            ("🟢 Agent 4 - Senior Adviseur", 'agent4')
        ]
        
        for title, key in agents:
            with st.expander(title):
                st.markdown(st.session_state.agent_outputs.get(key, "Geen output beschikbaar"))
    
    with tab3:
        st.markdown("### Download Opties")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download advies als tekstbestand
            advice_text = f"""JURIDISCH ADVIES
{'='*80}
Gegenereerd op: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Dossier: {st.session_state.casus_info['client_naam']} vs {st.session_state.casus_info['tegenpartij_naam']}
{'='*80}

{st.session_state.final_advice}
"""
            st.download_button(
                label="📥 Download Advies (TXT)",
                data=advice_text,
                file_name=f"juridisch_advies_{st.session_state.timestamp}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Download complete analyse als JSON
            complete_data = {
                'timestamp': st.session_state.timestamp,
                'casus_info': st.session_state.casus_info,
                'agent_outputs': st.session_state.agent_outputs,
                'final_advice': st.session_state.final_advice
            }
            
            st.download_button(
                label="📥 Download Complete Analyse (JSON)",
                data=json.dumps(complete_data, indent=2, ensure_ascii=False),
                file_name=f"complete_analyse_{st.session_state.timestamp}.json",
                mime="application/json"
            )
    
    # Nieuwe analyse knop
    st.divider()
    if st.button("🔄 Nieuwe Analyse", type="primary", use_container_width=True):
        for key in ['analyses_completed', 'agent_outputs', 'final_advice', 'casus_info', 'timestamp']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ================================
# MAIN APP
# ================================

def main():
    """Hoofdfunctie van de Streamlit app"""
    
    # Configureer Gemini API
    configure_gemini()
    
    # Toon header
    display_header()
    
    # Configureer sidebar
    display_sidebar()
    
    # App flow
    if not st.session_state.analyses_completed:
        # Verzamel casus informatie
        casus = get_casus_input()
        
        if casus:
            # Start analyse
            run_analysis(casus)
    else:
        # Toon resultaten
        display_results()

if __name__ == "__main__":
    main()
