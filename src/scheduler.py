from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import uvicorn
from api_user import app

scheduler = BackgroundScheduler()

def run_fastapi_app():
    uvicorn.run(app, host='127.0.0.1', port=8001)

scheduler.add_job(run_fastapi_app, CronTrigger(hour="*"))
scheduler.start()

if __name__ == '__main__':
    while True:
        pass
