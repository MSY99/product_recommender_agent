"""
RFP 파일의 타입에 따라 전처리 및 검색에 필요한 내용을 추출하는 모듈 모음
"""

import os
import io
import fitz
import tempfile
from docx import Document

from dotenv import load_dotenv

load_dotenv("/workspace/Dhq_chatbot/pdf2DB_Agent/.env")

from google import genai
import pathlib
from google.genai.types import Part

gemini_client = genai.Client()

def extract_text_from_pdf(file_buffer, client=gemini_client):
    # file_buffer: BytesIO or bytes

    if hasattr(file_buffer, "read"):
        pdf_bytes = file_buffer.read()
    else:
        pdf_bytes = file_buffer

    prompt_for_ocr = """이는 PDF 문서 입니다. 구조를 유지하면서 모든 텍스트 콘텐츠를 추출하세요. 테이블, 열, 헤더 및 모든 구조화된 콘텐츠에 특별히 주의하세요. 단락 구분 및 형식을 유지하세요.

이 문서 페이지에서 모든 텍스트 콘텐츠를 추출하세요. 

`테이블의 경우:
1. 마크다운 테이블 형식을 사용하여 테이블 구조를 유지하세요.
2. 모든 열 머리글 및 행 레이블을 보존하세요.
3. 숫자 데이터를 정확하게 기입하세요.
4. 하나의 칸이 2개 이상의 칸과 매치되는 경우를 유의해서 추출하세요.`

`다중 열 레이아웃의 경우:
1. 왼쪽에서 오른쪽으로 열을 처리하세요.
2. 서로 다른 열의 콘텐츠를 명확하게 구분하세요.`

`차트 및 그래프의 경우:
1. 차트 유형을 설명하세요.
2. 가시적인 축 레이블, 범례 및 데이터 포인트를 추출하세요.
3. 제목 또는 캡션을 추출하세요.`"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[ Part.from_bytes(
            data=pdf_bytes,
            mime_type='application/pdf',
        ),
        prompt_for_ocr]
    )

    extracted_txt = response.text

    with tempfile.NamedTemporaryFile('w', suffix='.txt', dir='/tmp', delete=False, encoding='utf-8') as tmpf:
        tmpf.write(extracted_txt)
        tmp_txt_path = tmpf.name

    return tmp_txt_path

def extract_text_from_docx(file_buffer):

    if hasattr(file_buffer, "read"):
        file_buffer.seek(0)
        doc = Document(file_buffer)
    else:
        doc = Document(io.BytesIO(file_buffer))

    text = "\n".join([para.text for para in doc.paragraphs])

    with tempfile.NamedTemporaryFile('w', suffix='.txt', dir='/tmp', delete=False, encoding="utf-8") as tmpf:
        tmpf.write(text)
        tmp_txt_path = tmpf.name
    return tmp_txt_path


def convert_rfp_to_RAG(tmp_txt_path, client=gemini_client):
    with open(tmp_txt_path, 'r', encoding='utf-8') as f:
        extract_text_from_pdf = f.read()

    prompt_for_convert = f"""아래 RFP 문서를 검토하여,

- 구매자가 원하는 제품(예: 스토리지, 서버)의 모든 필수 및 주요 요구조건(용량, 성능, 확장성, 프로토콜, 폼팩터, 보안, 가용성 등)을
- **자연스럽고 완결된 한 단락의 줄글(자연어 문장 모음)**로 요약해 주세요.

예시)

“이 프로젝트에서는 최소 2PB의 물리적 저장 용량과 SSD, HDD, 하이브리드 구성을 모두 지원하는 고성능 스토리지가 필요합니다. FC와 iSCSI 프로토콜을 모두 지원해야 하며, 자동 계층화, 데이터 중복 제거, 압축, AES-256 기반 암호화, 양방향 자동 페일오버 및 무중단 운영, 통합 관리 기능을 제공해야 합니다. 또한 다양한 운영체제와의 호환성, 스토리지 확장성, 고가용성 및 원격 관리 기능도 필수적으로 요구됩니다.”

이처럼,

- **RFP의 주요 요구조건을 하나의 자연스러운 줄글 단락**으로 작성해 주세요.
- 군더더기 없이, 실제 제품 비교에 바로 활용될 수 있도록 요약해 주세요.

[ RFP 문서 텍스트 ]
{extract_text_from_pdf}"""
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt_for_convert]
    )

    rfp_search_term = response.text

    return rfp_search_term.strip()

