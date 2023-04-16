import email
import openai
import pandas as pd
import os
import re
import smtplib
import streamlit as st
import time

from email.message import EmailMessage
from email.utils import formatdate
from os.path import join
from scipy import spatial
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from typing import List


# Credentials (--> later to ENV)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMAIL = os.environ["EMAIL"]
PASSWORD = os.environ["PASSWORD"]
SMTP_SERVER = os.environ["SMTP_SERVER"]

# Set openai api key
openai.api_key = OPENAI_API_KEY

# Path to webdriver
cwd_path = os.path.abspath(os.getcwd())
driver_path = join(cwd_path, 'assets/firefox')
st.markdown(driver_path)

# URL of HAD page
HAD_URL = 'https://www.had.de/onlinesuche_einfach.html'

# Global Streamlit settings
st.set_page_config(layout="wide", page_title="HAD Screening")

with open("./css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Some self-marekting, because -- why not?
st.markdown(
    """
        <style>
            [data-testid="stHeader"]::before {
                content: "ai4ki";
                font-family: Arial, sans-serif;
                font-weight: bold;
                font-size: 40px;
                color: #245606;
                position: relative;
                left: 30px;
                top: 10px;
            }
        </style>
        """,
    unsafe_allow_html=True,
)

# Define page layout
cols = st.columns([4, 1, 6, 1, 6])

# Load model prompts
with open("./assets/positive_prompt_de.txt", "r", encoding="utf-8") as f:
    prompt = f.read()


def get_had_table():

    had_table = pd.DataFrame()
    error_string = ""

    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Firefox(driver_path, options=options)
        driver.get(HAD_URL)

        select_element = Select(driver.find_element(By.NAME, "L_CAT"))
        select_element.select_by_value("SQLB")

        radio_button = driver.find_element(By.XPATH, '//input[@type="radio" and @value="500"]')
        radio_button.click()

        submit = driver.find_element(By.CLASS_NAME, "submit")
        submit.click()

        wait = WebDriverWait(driver, 10)
        content = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))

        table_html = content.get_attribute('outerHTML')
        pattern = r'<div class="small">.*?</div>'  # because is scrambles table
        table_clean = re.sub(pattern, '', table_html)

        had_table = pd.read_html(table_clean, encoding='utf-8', header=0)[0]
        had_table.dropna(axis=0, how='all', inplace=True)
        had_table.drop(had_table.columns[[0]], axis=1, inplace=True)
        had_table.reset_index(drop=True, inplace=True)
        had_table.rename(columns={'VerfahrenLeistung': 'Ausschreibung'}, inplace=True)

    except:
        error_string = "ERROR: Access to HAD database failed :("

    return had_table, error_string


def send_email(subject, address, body):
    msg = EmailMessage()
    msg['Date'] = email.utils.formatdate()
    msg['Message-ID'] = email.utils.make_msgid(domain='ai4ki.org')
    msg['From'] = EMAIL
    msg['To'] = address
    msg['Subject'] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_SERVER, 465) as smtp:
        smtp.login(EMAIL, PASSWORD)
        smtp.send_message(msg)


def distances_from_embeddings(
        query_embedding: List[float],
        embeddings: List[List[float]],
        distance_metric="cosine",
) -> List[List]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def evaluate_calls(df, query):

    query_embedding = openai.Embedding.create(input=query, engine='text-embedding-ada-002')['data'][0]['embedding']

    df['embeddings'] = df.Ausschreibung.apply(
        lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

    df['distances'] = distances_from_embeddings(query_embedding, df['embeddings'].values, distance_metric='cosine')

    return df


def keyword_check(word):

    with open("./assets/negative_keywords.txt", "r", encoding="utf-8") as file:
        negative_keywords = file.read().lower().split(",")

    keyword_flag = True
    for keyword in negative_keywords:
        if keyword.strip() in word.lower():
            keyword_flag = False

    return keyword_flag


with cols[0]:

    if "table" not in st.session_state:
        table_as_df, had_error = get_had_table()
        st.session_state.table = table_as_df
        st.session_state.error = had_error

    st.markdown("#### Institutsprofil")
    st.markdown(f"{prompt}")

    if st.session_state.error:
        st.markdown(f"### Ups, das ist leider was schief gegangen :((( | System message: {st.session_state.error}")
    else:
        n_calls = len(st.session_state.table)
        st.markdown(f"**{n_calls} Ausschreibungen in HAD gefunden**")
        top_k = st.slider("**Maximale Anzahl an Treffern:**", 1, 10, 3)
        mail_results_to = st.text_input(label="**Ergebnisse mailen an:**", placeholder="person@example.org")

with cols[2]:
    st.markdown(f"### Aktuelle HAD-Ausschreibungen")
    st.dataframe(st.session_state.table)
    st.markdown("")
    had_analyze = st.button("HAD auswerten")

with cols[4]:
    if had_analyze:
        alert = "**Dieser Newsletter enthält vermutlich keine passenden Ausschreibungen!**"
        st.markdown("### Auswertungsergebnis")
        with st.spinner(f"Auswertung von HAD läuft..."):
            result_table = evaluate_calls(st.session_state.table, prompt)

            hit_count = 1
            result = ""
            for idx, rows in result_table.sort_values('distances', ascending=True).iterrows():
                if hit_count == 1:
                    if rows.distances > 0.18:
                        result += alert + 2 * "\n"
                        st.markdown(alert)

                if keyword_check(rows.Ausschreibung):
                    client_data = f"**HAD-Ausschreibung {idx}**<br>{rows.Ausschreibung}"
                    st.markdown(
                        f"{client_data}<br>*Übereinstimmung: {(1.0 - rows.distances) * 100:.2f} %*",
                        unsafe_allow_html=True
                    )
                    result += f"{client_data}\n\n"

                if hit_count == top_k:
                    break
                hit_count += 1

            if mail_results_to:
                if re.match(r"[^@]+@[^@]+\.[^@]+", mail_results_to):
                    header = f"Die Auswertung von HAD hat folgendes ergeben:"
                    email_body = header + 2 * "\n" + result
                    send_email('Newsletter-Auswertung', mail_results_to, email_body)
                    time.sleep(5)
