import json
from typing import Any, Dict, List
import pandas as pd
import os
from pydantic import BaseModel, Field
from sqlglot import parse_one, exp

try:
    import asyncpg
except ImportError:
    asyncpg = None

class SQLInput(BaseModel):
    query: str = Field(..., description="The SQL SELECT query to execute against the PostgreSQL database.")

class SQLOutput(BaseModel):
    success: bool = Field(..., description="Indicates if the query was successfully validated and executed.")
    data: str = Field(..., description="JSON string representation of the retrieved records or the error message.")

class SQLEngine:
    """Async PG client with sqlglot AST-based RBAC validator."""
    def __init__(self, dsn: str = None):
        self.dsn = dsn or os.getenv("DB_READONLY_URI", "postgresql://naminsight_reader:reader_secure_pass@localhost:5432/naminsight")
        if asyncpg is None:
            raise ImportError("Please install asyncpg: pip install asyncpg")
        
    def validate_query(self, sql_string: str) -> bool:
        """Validate AST for strict SELECT-only compliance."""
        try:
            parsed = parse_one(sql_string, read="postgres")
        except Exception as e:
            raise ValueError(f"SQL parsing error: {str(e)}")
            
        if not isinstance(parsed, exp.Select):
            raise ValueError(f"Safety Violation: Only SELECT statements are allowed. Found: {type(parsed).__name__}")
            
        for node in parsed.find_all(exp.Command):
            raise ValueError("Safety Violation: Commands are not allowed.")

        for node in parsed.find_all((exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Execute)):
            raise ValueError(f"Safety Violation: DML/DDL node found ({type(node).__name__}).")

        return True

    async def execute(self, query: str) -> SQLOutput:
        """Run safe SELECT & return JSON schema."""
        try:
            self.validate_query(query)
            
            conn = await asyncpg.connect(self.dsn)
            try:
                records = await conn.fetch(query)
                results = [dict(record) for record in records]
                
                df = pd.DataFrame(results)
                json_data = df.to_json(orient="records")
                
                return SQLOutput(success=True, data=json_data)
            finally:
                await conn.close()
                
        except ValueError as ve:
            return SQLOutput(success=False, data=json.dumps({"error": str(ve)}))
        except Exception as e:
            return SQLOutput(success=False, data=json.dumps({"error": f"Database error: {str(e)}"}))
            
    async def run(self, **kwargs) -> str:
        """Tool runner entrypoint."""
        req = SQLInput(**kwargs)
        out = await self.execute(query=req.query)
        return out.model_dump_json()
