from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    "update_sol_price": {
        "task": "update_data.update_data",
        "schedule": crontab(hour="*/12"),  # Runs every 12 hours
    },
}

CELERY_TIMEZONE = "UTC"  # Or your preferred timezone