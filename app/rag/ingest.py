from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from transformers import AutoTokenizer

from app.config import Settings, get_settings
from app.rag.embeddings import get_embeddings
from app.rag.schemas import Chunk

log = logging.getLogger("ingest")

ARCHIVOS: list[tuple[str, str]] = [
    ("copa_libertadores.md", "torneos"),
    ("formaciones_tacticas.pdf", "tacticas"),
    ("mundiales_argentina.json", "selecciones"),
    ("regla_fuera_de_juego.md", "reglamento"),
    ("var_videoarbitraje.pdf", "reglamento"),
]


def _read_md(path: Path, categoria: str) -> list[Document]:
    return [Document(page_content=path.read_text(encoding="utf-8"),
                     metadata={"source": path.name, "categoria": categoria, "page": 1})]


def _read_pdf(path: Path, categoria: str) -> list[Document]:
    reader = PdfReader(str(path))
    return [
        Document(page_content=page.extract_text() or "",
                 metadata={"source": path.name, "categoria": categoria, "page": i})
        for i, page in enumerate(reader.pages, start=1)
    ]


def _read_json(path: Path, categoria: str) -> list[Document]:
    datos = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(page_content=f"{sec['subtitulo']} - {sec['contenido']}",
                 metadata={"source": path.name, "categoria": categoria, "page": i})
        for i, sec in enumerate(datos["secciones"], start=1)
    ]


def load_documents(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for nombre, categoria in ARCHIVOS:
        path = data_dir / nombre
        if path.suffix == ".md":
            docs += _read_md(path, categoria)
        elif path.suffix == ".pdf":
            docs += _read_pdf(path, categoria)
        elif path.suffix == ".json":
            docs += _read_json(path, categoria)
    return docs


def _clean(text: str) -> str:
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def build_chunks(settings: Settings) -> list[Chunk]:
    docs = load_documents(Path(settings.data_dir))
    for d in docs:
        d.page_content = _clean(d.page_content)
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return [
        Chunk(text=d.page_content, source=d.metadata["source"],
              categoria=d.metadata["categoria"], page=d.metadata["page"])
        for d in splitter.split_documents(docs)
        if d.page_content.strip()
    ]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def ensure_index(settings: Settings) -> Pinecone:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not pc.has_index(settings.pinecone_index):
        log.info("creando índice %s ...", settings.pinecone_index)
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return pc


def vectors_in_namespace(pc: Pinecone, settings: Settings) -> int:
    stats = pc.Index(settings.pinecone_index).describe_index_stats()
    return stats.get("namespaces", {}).get(settings.pinecone_namespace, {}).get("vector_count", 0)


async def upsert(chunks: list[Chunk], settings: Settings) -> None:
    docs = [Document(page_content=c.text, metadata=c.model_dump(exclude={"text"})) for c in chunks]
    await PineconeVectorStore.afrom_documents(
        documents=docs,
        embedding=get_embeddings(),
        index_name=settings.pinecone_index,
        namespace=settings.pinecone_namespace,
        pinecone_api_key=settings.pinecone_api_key,
    )


async def run_ingest(force: bool = False) -> int:
    settings = get_settings()
    chunks_path = Path(settings.chunks_file)

    chunks = build_chunks(settings)
    save_chunks(chunks, chunks_path)
    log.info("%d chunks generados -> %s", len(chunks), chunks_path)

    pc = ensure_index(settings)
    existing = vectors_in_namespace(pc, settings)
    if existing and not force:
        log.info("namespace %s ya tiene %d vectores, no se resube (--force para forzar)",
                 settings.pinecone_namespace, existing)
        return existing

    log.info("subiendo %d chunks a Pinecone...", len(chunks))
    await upsert(chunks, settings)
    log.info("listo")
    return len(chunks)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run_ingest(force="--force" in sys.argv))
