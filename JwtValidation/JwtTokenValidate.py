from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError, ExpiredSignatureError
from starlette.middleware.sessions import SessionMiddleware
from ApplicationConstants import AppConstants
# -------------------------
# CONFIG (MATCH SPRING)
# -------------------------

ALGORITHM = "HS256"

app = FastAPI()
security = HTTPBearer(bearerFormat="JWT")


app.add_middleware(
    SessionMiddleware,
    secret_key="neshauthkey"
)

def rebuild_token_like_spring(jwt_token: str) -> str:
    """
    Replicates Spring logic:
    extractToken = payload.substring(3, len-3)
    oldtoken = header.payload.signature
    """
    try:
        header, payload, signature = jwt_token.split(".")
        modified_payload = payload[3:-3]
        rebuilt = f"{header}.{modified_payload}.{signature}"
        return rebuilt
    except Exception:
        return None


def validate_token(auth_token: str) -> bool:
    """
    Validates the modified token.
    Replicates Java: Jwts.parser().setSigningKey(secret).parseClaimsJws(authToken)
    """
    try:
        jwt.decode(
            auth_token,
            AppConstants.SECRET_KEY,
            algorithms=["HS512"],
            options={"verify_signature": False}
        )

        return True
    except ExpiredSignatureError:
        raise ExpiredSignatureError("Token has expired")
    except (JWTError) as ex:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="INVALID_CREDENTIALS"
        ) from ex


def get_all_claim_details_from_token(token: str) -> dict:
    """
    Extracts all claims from the token.
    Replicates Java: jwtTokenUtil.getAllClaimDetailsFromToken(oldtoken)
    """
    try:
       claims = jwt.decode(
            token,
            AppConstants.SECRET_KEY,
            algorithms=["HS512"],
            options={"verify_signature": False}
        )
       return claims
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="INVALID_CREDENTIALS"
        ) from ex

def jwt_filter(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Replicates Spring filter logic for JWT validation and claims extraction.
    """
    try:
        jwt_token = credentials.credentials
        
        if jwt_token:
            # Rebuild token like Spring
            old_token = rebuild_token_like_spring(jwt_token)
            if not old_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token format"
                )
            
            # Validate the modified token
            if validate_token(old_token):
                # Extract all claims from the modified token
                claims = get_all_claim_details_from_token(old_token)
                
                # Store claims in request state (similar to request.setAttribute)
                request.state.user_id = claims.get("userId")
                request.state.role_id = claims.get("roleId")
                request.state.user_role = claims.get("userRole")
                request.state.user_org_type = claims.get("userOrgType")
                request.state.reseller_id = claims.get("resellerId")
                request.state.customer_id = claims.get("customerId")
                request.state.org_id = claims.get("orgId")
                request.state.logo_name = claims.get("logoName")
                request.state.user_name = claims.get("userName")
                request.state.vehicle_label = claims.get("vehicleLabel")
                
                # # Store claims in session
                # request.session["claims"] = claims
                
                return claims
        
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token not found")

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


@app.get("/service")
def secured_service(
    request: Request,
    claims: dict = Depends(jwt_filter)
):
    """
    Example endpoint that uses the JWT filter.
    """
    return {
        "message": "Access granted",
        "userId": request.state.user_id,
        "userName": request.state.user_name,
        "claims": claims
    }
