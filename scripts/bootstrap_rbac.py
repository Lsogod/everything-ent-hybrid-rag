from app.services.rbac import RBACService


if __name__ == "__main__":
    service = RBACService()
    service.bootstrap_defaults()
    print("rbac bootstrap completed")
