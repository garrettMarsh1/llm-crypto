#!/usr/bin/env python3
"""
System Validation Script
Tests all components before training
"""

import sys
import os
import json
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("=== Testing Imports ===")
    
    try:
        import torch
        print(f"OK PyTorch: {torch.__version__}")
        print(f"OK CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"OK CUDA version: {torch.version.cuda}")
            print(f"OK GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"FAIL PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"OK Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"FAIL Transformers import failed: {e}")
        return False
    
    try:
        import chromadb
        print(f"OK ChromaDB: {chromadb.__version__}")
    except ImportError as e:
        print(f"FAIL ChromaDB import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print(f"OK Sentence Transformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"FAIL Sentence Transformers import failed: {e}")
        return False
    
    try:
        from memory.agent_memory_system import AgentMemorySystem
        print("OK Memory system imports successfully")
    except ImportError as e:
        print(f"FAIL Memory system import failed: {e}")
        return False
    except Exception as e:
        print(f"FAIL Memory system import error: {e}")
        return False
    
    try:
        from memory.agent_lifecycle_manager import AgentLifecycleManager
        print("OK Agent lifecycle manager imports successfully")
    except ImportError as e:
        print(f"FAIL Agent lifecycle manager import failed: {e}")
        return False
    except Exception as e:
        print(f"FAIL Agent lifecycle manager import error: {e}")
        return False
    
    return True

def test_training_data():
    """Test training data availability and quality"""
    print("\n=== Testing Training Data ===")
    
    data_path = "training_data/combined_training_data.json"
    if not os.path.exists(data_path):
        print(f"FAIL Training data not found: {data_path}")
        return False
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"OK Training data loaded: {len(data)} samples")
        
        # Check data structure
        sample = data[0]
        required_keys = ['symbol', 'signal', 'confidence', 'reasoning', 'price_data', 'technical_indicators', 'market_context']
        
        for key in required_keys:
            if key not in sample:
                print(f"FAIL Missing key in training data: {key}")
                return False
        
        print("OK Training data structure is valid")
        
        # Check signal distribution
        signals = [s['signal'] for s in data]
        signal_counts = {}
        for signal in signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        print("OK Signal distribution:")
        for signal, count in signal_counts.items():
            print(f"  {signal}: {count} ({count/len(data)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"FAIL Training data validation failed: {e}")
        return False

def test_memory_system():
    """Test memory system functionality"""
    print("\n=== Testing Memory System ===")
    
    try:
        from memory.agent_memory_system import AgentMemorySystem
        
        # Create test config
        config = {
            'memory': {
                'db_path': './memory/test_memories.db',
                'max_tokens_per_agent': 1000,
                'max_context_tokens': 100,
                'compression_threshold': 50
            }
        }
        
        # Initialize memory system
        memory_system = AgentMemorySystem(config)
        print("OK Memory system initialized")
        
        # Create test agent
        agent_id = memory_system.create_agent()
        print(f"OK Agent created: {agent_id}")
        
        # Add test memory
        memory_id = memory_system.add_memory(
            agent_id=agent_id,
            content="Test memory for BTC analysis",
            memory_type="analysis",
            importance=0.8,
            tags=["test", "BTC"]
        )
        print(f"OK Memory added: {memory_id}")
        
        # Get context
        context, tokens = memory_system.get_agent_context(agent_id, "Test query")
        print(f"OK Context retrieved: {tokens} tokens")
        
        # Cleanup
        memory_system.close()
        if os.path.exists('./memory/test_memories.db'):
            os.remove('./memory/test_memories.db')
        print("OK Memory system test completed")
        
        return True
        
    except Exception as e:
        print(f"FAIL Memory system test failed: {e}")
        return False

def test_model_loading():
    """Test model loading capabilities"""
    print("\n=== Testing Model Loading ===")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        from transformers import BitsAndBytesConfig
        
        # Check if prepare_model_for_int8_training is available
        try:
            from peft import prepare_model_for_int8_training
        except ImportError:
            # Use alternative for newer PEFT versions
            from peft import prepare_model_for_kbit_training as prepare_model_for_int8_training
        
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Testing model: {model_name}")
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("OK Tokenizer loaded successfully")
        
        # Test model loading with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype="float16"
        )
        print("OK Model loaded with 4-bit quantization")
        
        # Test LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        print("OK LoRA configuration applied")
        
        # Test generation with strict schema
        test_prompt = (
            "You are an expert crypto trading assistant.\n"
            "Analyze the BTC market data below and respond EXACTLY in this format:\n"
            "SIGNAL: <BUY|SELL|HOLD>\n"
            "CONFIDENCE: <0.00-1.00>\n"
            "REASONING: <1-3 concise sentences>.\n\n"
            "Market Data: BTC-USD."
        )
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.35,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(test_prompt):].strip()
        print(f"OK Model generation test successful")
        print(f"  Prompt: {test_prompt}")
        print(f"  Full Response:")
        print(f"  {'='*60}")
        print(f"  {generated_text}")
        print(f"  {'='*60}")
        
        return True
        
    except Exception as e:
        print(f"FAIL Model loading test failed: {e}")
        return False

def test_config_files():
    """Test configuration files"""
    print("\n=== Testing Configuration Files ===")
    
    config_files = [
        "training_config.yaml",
        "memory_config.yaml", 
        "rlhf_config.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"OK {config_file} exists")
        else:
            print(f"FAIL {config_file} missing")
            return False
    
    return True

def main():
    """Run all validation tests"""
    print("Starting System Validation...")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Training Data", test_training_data),
        ("Memory System", test_memory_system),
        ("Model Loading", test_model_loading),
        ("Config Files", test_config_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"PASS: {test_name} test PASSED")
            else:
                print(f"FAIL: {test_name} test FAILED")
        except Exception as e:
            print(f"ERROR: {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All systems ready for training!")
        return True
    else:
        print("WARNING: Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
