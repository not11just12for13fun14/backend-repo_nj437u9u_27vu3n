import os
from datetime import datetime, time
from typing import List, Optional, Literal, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bson import ObjectId

from database import db, create_document, get_documents

app = FastAPI(title="Hall Booking Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
# -----------------------------
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        try:
            return ObjectId(str(v))
        except Exception:
            raise ValueError("Invalid ObjectId")


def oid(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    doc["id"] = str(doc.pop("_id"))
    for k, v in list(doc.items()):
        if isinstance(v, ObjectId):
            doc[k] = str(v)
        if isinstance(v, list):
            doc[k] = [str(x) if isinstance(x, ObjectId) else x for x in v]
    return doc


# -----------------------------
# Schemas
# -----------------------------
class Hall(BaseModel):
    name: str
    capacity: int = Field(ge=1)
    equipment: List[str] = Field(default_factory=list)


class BookingStatus(BaseModel):
    status: Literal["pending", "confirmed", "rejected", "cancelled"]


class AuditEntry(BaseModel):
    by_name: str
    role: str
    action: Literal["requested", "approved", "rejected", "cancelled", "updated"]
    comment: Optional[str] = None
    at: datetime = Field(default_factory=datetime.utcnow)


class Booking(BaseModel):
    hall_id: str
    requestor_id: Optional[str] = None
    requestor_name: str
    requestor_email: Optional[str] = None
    date: str  # YYYY-MM-DD
    start: str  # HH:MM
    end: str  # HH:MM
    event_name: str
    purpose: Optional[str] = None
    attendance: Optional[int] = None
    attachments: List[str] = Field(default_factory=list)
    status: Literal["pending", "confirmed", "rejected", "cancelled"] = "pending"
    audit_trail: List[AuditEntry] = Field(default_factory=list)
    approver_flow: List[str] = Field(default_factory=lambda: ["HOD", "Admin"])  # preview of steps


class BookingCreate(BaseModel):
    hall_id: str
    requestor_id: Optional[str] = None
    requestor_name: str
    requestor_email: Optional[str] = None
    date: str
    start: str
    end: str
    event_name: str
    purpose: Optional[str] = None
    attendance: Optional[int] = None
    attachments: List[str] = Field(default_factory=list)


class Decision(BaseModel):
    by_name: str
    role: str
    comment: Optional[str] = None


# -----------------------------
# Utility
# -----------------------------

def parse_hhmm(s: str) -> time:
    try:
        hh, mm = s.split(":")
        return time(hour=int(hh), minute=int(mm))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid time format, expected HH:MM")


def overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    s1, e1 = parse_hhmm(a_start), parse_hhmm(a_end)
    s2, e2 = parse_hhmm(b_start), parse_hhmm(b_end)
    return s1 < e2 and s2 < e1


# -----------------------------
# Seed sample halls (idempotent)
# -----------------------------
@app.on_event("startup")
def seed_data():
    if db is None:
        return
    if "hall" not in db.list_collection_names():
        db.create_collection("hall")
    if db["hall"].count_documents({}) == 0:
        db["hall"].insert_many([
            {"name": "Alpha Hall", "capacity": 100, "equipment": ["Projector", "PA System"]},
            {"name": "Beta Auditorium", "capacity": 300, "equipment": ["Projector", "Stage Lighting", "Recording"]},
            {"name": "Gamma Room", "capacity": 40, "equipment": ["TV Display"]},
        ])
    if "booking" not in db.list_collection_names():
        db.create_collection("booking")
    db["booking"].create_index([("hall_id", 1), ("date", 1), ("start", 1), ("end", 1)])


# -----------------------------
# Health and schema endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Hall Booking Backend running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Set"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# -----------------------------
# Halls & Availability
# -----------------------------
class HallFilter(BaseModel):
    date: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    min_capacity: Optional[int] = None
    equipment: List[str] = Field(default_factory=list)


@app.post("/halls/search")
def search_halls(filters: HallFilter):
    q: Dict[str, Any] = {}
    if filters.min_capacity:
        q["capacity"] = {"$gte": filters.min_capacity}
    if filters.equipment:
        q["equipment"] = {"$all": filters.equipment}

    halls = [serialize_doc(h) for h in db["hall"].find(q)] if db else []

    # attach availability status if time window provided
    for h in halls:
        status = "free"
        color = "green"
        if filters.date and filters.start and filters.end and db:
            # find overlapping bookings for this hall on that date
            overlapping = list(
                db["booking"].find(
                    {
                        "hall_id": h["id"],
                        "date": filters.date,
                        "status": {"$in": ["pending", "confirmed"]},
                    }
                )
            )
            conflict = any(overlaps(filters.start, filters.end, b["start"], b["end"]) for b in overlapping)
            if conflict:
                if any(b.get("status") == "confirmed" for b in overlapping if overlaps(filters.start, filters.end, b["start"], b["end"])):
                    status, color = "booked", "red"
                else:
                    status, color = "pending", "yellow"
        h["availability"] = {"status": status, "color": color}
    return {"results": halls}


# -----------------------------
# Bookings
# -----------------------------
@app.post("/bookings")
def create_booking(payload: BookingCreate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    # Check hall exists
    hall = db["hall"].find_one({"_id": oid(payload.hall_id)}) if ObjectId.is_valid(payload.hall_id) else db["hall"].find_one({"_id": oid(payload.hall_id)})
    if not hall:
        raise HTTPException(status_code=404, detail="Hall not found")

    # Prevent double booking (conflict with confirmed or pending)
    conflicts = list(
        db["booking"].find(
            {
                "hall_id": str(hall["_id"]),
                "date": payload.date,
                "status": {"$in": ["pending", "confirmed"]},
            }
        )
    )
    if any(overlaps(payload.start, payload.end, b["start"], b["end"]) for b in conflicts):
        raise HTTPException(status_code=409, detail="Time slot not available")

    booking = Booking(
        hall_id=str(hall["_id"]),
        requestor_id=payload.requestor_id,
        requestor_name=payload.requestor_name,
        requestor_email=payload.requestor_email,
        date=payload.date,
        start=payload.start,
        end=payload.end,
        event_name=payload.event_name,
        purpose=payload.purpose,
        attendance=payload.attendance,
        attachments=payload.attachments,
        audit_trail=[AuditEntry(by_name=payload.requestor_name, role="Requestor", action="requested")],
    ).model_dump()

    new_id = db["booking"].insert_one(booking).inserted_id
    saved = db["booking"].find_one({"_id": new_id})
    return serialize_doc(saved)


@app.get("/bookings")
def list_bookings(requestor_id: Optional[str] = None, status: Optional[str] = None):
    if db is None:
        return {"results": []}
    q: Dict[str, Any] = {}
    if requestor_id:
        q["requestor_id"] = requestor_id
    if status:
        q["status"] = status
    bookings = [serialize_doc(b) for b in db["booking"].find(q).sort("date", 1)]
    return {"results": bookings}


@app.get("/bookings/{booking_id}")
def get_booking(booking_id: str):
    b = db["booking"].find_one({"_id": oid(booking_id)}) if db else None
    if not b:
        raise HTTPException(status_code=404, detail="Not found")
    return serialize_doc(b)


@app.post("/bookings/{booking_id}/approve")
def approve_booking(booking_id: str, decision: Decision):
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")
    b = db["booking"].find_one({"_id": oid(booking_id)})
    if not b:
        raise HTTPException(status_code=404, detail="Not found")
    if b.get("status") in ["rejected", "cancelled", "confirmed"]:
        raise HTTPException(status_code=400, detail="Booking already finalized")

    # Determine next step in approver_flow
    flow: List[str] = b.get("approver_flow", ["HOD", "Admin"])
    trail: List[Dict[str, Any]] = b.get("audit_trail", [])
    approvals_done = [t for t in trail if t.get("action") == "approved"]
    next_step = flow[len(approvals_done)] if len(approvals_done) < len(flow) else None

    trail.append(
        AuditEntry(by_name=decision.by_name, role=decision.role, action="approved", comment=decision.comment).model_dump()
    )

    new_status = "confirmed" if next_step is None else "pending"
    update = {"$set": {"audit_trail": trail, "status": new_status}}
    db["booking"].update_one({"_id": oid(booking_id)}, update)

    updated = db["booking"].find_one({"_id": oid(booking_id)})
    return serialize_doc(updated)


@app.post("/bookings/{booking_id}/reject")
def reject_booking(booking_id: str, decision: Decision):
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")
    if not decision.comment:
        raise HTTPException(status_code=400, detail="Reason/Comment is required for rejection")
    b = db["booking"].find_one({"_id": oid(booking_id)})
    if not b:
        raise HTTPException(status_code=404, detail="Not found")
    if b.get("status") in ["rejected", "cancelled", "confirmed"]:
        raise HTTPException(status_code=400, detail="Booking already finalized")

    trail: List[Dict[str, Any]] = b.get("audit_trail", [])
    trail.append(
        AuditEntry(by_name=decision.by_name, role=decision.role, action="rejected", comment=decision.comment).model_dump()
    )

    db["booking"].update_one({"_id": oid(booking_id)}, {"$set": {"audit_trail": trail, "status": "rejected"}})
    updated = db["booking"].find_one({"_id": oid(booking_id)})
    return serialize_doc(updated)


# Approver queue
@app.get("/approvals/queue")
def approvals_queue(role: Optional[str] = None):
    if db is None:
        return {"results": []}
    q = {"status": "pending"}
    if role:
        # If role provided, include those pending with approvals_done < index of role in flow
        pipeline = [
            {"$match": q},
            {"$addFields": {"approvals_done": {"$size": {"$filter": {"input": "$audit_trail", "as": "t", "cond": {"$eq": ["$$t.action", "approved"]}}}}}},
        ]
        items = list(db["booking"].aggregate(pipeline))
        # Filter in Python for simplicity: next step role equals provided role
        results = []
        for it in items:
            flow = it.get("approver_flow", ["HOD", "Admin"])
            idx = it.get("approvals_done", 0)
            next_step = flow[idx] if idx < len(flow) else None
            if next_step == role:
                results.append(serialize_doc(it))
        return {"results": results}
    else:
        results = [serialize_doc(b) for b in db["booking"].find(q)]
        return {"results": results}


# -----------------------------
# Simple schema exposure for tooling
# -----------------------------
@app.get("/schema")
def schema_overview():
    return {
        "hall": Hall.model_json_schema(),
        "booking": Booking.model_json_schema(),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
