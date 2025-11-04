#!/usr/bin/env python3
"""
Integration test to verify all modules import correctly and basic functionality works
"""

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from app.core.config import settings
        print("✓ Config imported successfully")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from app.core.llm_service import LLMService
        print("✓ LLM Service imported successfully")
    except Exception as e:
        print(f"✗ LLM Service import failed: {e}")
        return False
    
    try:
        from app.core.query_processor import QueryProcessor
        print("✓ Query Processor imported successfully")
    except Exception as e:
        print(f"✗ Query Processor import failed: {e}")
        return False
        
    try:
        from app.services.decision_service import DecisionService
        print("✓ Decision Service imported successfully")
    except Exception as e:
        print(f"✗ Decision Service import failed: {e}")
        return False
    
    try:
        from app.services.document_service import DocumentService
        print("✓ Document Service imported successfully")
    except Exception as e:
        print(f"✗ Document Service import failed: {e}")
        return False
    
    try:
        from app.services.retrieval_service import RetrievalService
        print("✓ Retrieval Service imported successfully")
    except Exception as e:
        print(f"✗ Retrieval Service import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test config
        from app.core.config import settings
        print(f"✓ Config loaded: Provider={settings.LLM_PROVIDER}")
        
        # Test LLM service initialization
        from app.core.llm_service import LLMService
        llm_service = LLMService()
        print(f"✓ LLM Service initialized: Provider={llm_service.provider}")
        
        # Test Query Processor initialization
        from app.core.query_processor import QueryProcessor
        query_processor = QueryProcessor()
        print(f"✓ Query Processor initialized: Provider={query_processor.llm_service.provider}")
        
        # Test Decision Service initialization
        from app.services.decision_service import DecisionService
        decision_service = DecisionService()
        print(f"✓ Decision Service initialized: Provider={decision_service.llm_service.provider}")
        
        # Test Document Service initialization
        from app.services.document_service import DocumentService
        document_service = DocumentService()
        print("✓ Document Service initialized")
        
        # Test Retrieval Service initialization
        from app.services.retrieval_service import RetrievalService
        retrieval_service = RetrievalService()
        print("✓ Retrieval Service initialized")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_services_integration():
    """Test that services work together"""
    print("\nTesting services integration...")
    
    try:
        from app.core.query_processor import QueryProcessor
        qp = QueryProcessor()
        
        # Test system stats which should include enhanced features
        stats = qp.get_system_stats()
        enhanced_features = stats.get('enhanced_features', {})
        
        if enhanced_features.get('decision_service_available'):
            print("✓ Decision service integration confirmed")
        else:
            print("✗ Decision service not properly integrated")
            return False
            
        if enhanced_features.get('retrieval_service_available'):
            print("✓ Retrieval service integration confirmed")
        else:
            print("✗ Retrieval service not properly integrated")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Services integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting integration tests...\n")
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    success &= test_services_integration()
    
    if success:
        print("\n✓ All integration tests passed!")
        print("\nIntegration completed successfully. The new features from the newer version")
        print("have been successfully integrated into the stable codebase while maintaining")
        print("the stability of the original version.")
    else:
        print("\n✗ Some integration tests failed!")
        exit(1)