def add_health_endpoint(app):
    @app.get("/health")
    async def health():
        return {"status": "ok"}
