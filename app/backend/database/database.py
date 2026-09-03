from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/research_assistant"

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with async_session() as session:
        yield session
