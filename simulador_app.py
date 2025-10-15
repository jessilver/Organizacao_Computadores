import streamlit as st
import pandas as pd
import struct
import random
from copy import deepcopy

# ===================================================================
# PARTE A: LÓGICA DO SIMULADOR (Classes, Parsers, Execução)
# ===================================================================

class SimpleMemory:
    def __init__(self, size_in_words=64): self.size = size_in_words; self.mem = [0] * size_in_words
    def load_word(self, address):
        if 0 <= address < self.size: return self.mem[address]
        raise IndexError(f"Endereço de memória inválido: {address}")
    def store_word(self, address, value):
        if 0 <= address < self.size: self.mem[address] = value & 0xFFFFFFFF
        else: raise IndexError(f"Endereço de memória inválido: {address}")

class RegisterFile:
    def __init__(self): self.regs = [0] * 32
    def read(self, reg_num):
        if 0 <= reg_num < 32: return self.regs[reg_num]
        raise IndexError(f"Número de registrador inválido: x{reg_num}")
    def write(self, reg_num, value):
        if reg_num == 0: return
        if 0 < reg_num < 32: self.regs[reg_num] = value & 0xFFFFFFFF
        else: raise IndexError(f"Número de registrador inválido: x{reg_num}")

class Cache:
    def __init__(self, num_sets, main_memory, replacement_policy='lru'):
        self.num_sets = num_sets; self.main_memory = main_memory; self.replacement_policy = replacement_policy
        self.sets = [[self._create_cache_line(), self._create_cache_line()] for _ in range(num_sets)]
        self.hits, self.misses = 0, 0
    def _create_cache_line(self): return {'valid': 0, 'tag': 0, 'data': 0, 'lru': 0}
    def _get_address_parts(self, address): return address % self.num_sets, address // self.num_sets
    def read(self, address):
        index, tag = self._get_address_parts(address); target_set = self.sets[index]
        for i, line in enumerate(target_set):
            if line['valid'] == 1 and line['tag'] == tag:
                self.hits += 1; target_set[0]['lru'], target_set[1]['lru'] = (1-i), i
                return line['data'], 'hit'
        self.misses += 1; data = self.main_memory.load_word(address)
        way_to_replace = next((i for i, line in enumerate(target_set) if line['lru'] != i), 0) if self.replacement_policy == 'lru' else random.choice([0, 1])
        target_set[way_to_replace].update({'valid': 1, 'tag': tag, 'data': data})
        target_set[0]['lru'], target_set[1]['lru'] = (1-way_to_replace), way_to_replace
        return data, 'miss'
    def write(self, address, data):
        index, tag = self._get_address_parts(address); target_set = self.sets[index]; self.main_memory.store_word(address, data)
        for i, line in enumerate(target_set):
            if line['valid'] == 1 and line['tag'] == tag:
                self.hits += 1; line['data'] = data; target_set[0]['lru'], target_set[1]['lru'] = (1-i), i
                return None, 'hit'
        self.misses += 1
        return None, 'miss'
    def get_stats(self):
        total = self.hits + self.misses
        if total == 0: return 0.0, 0.0, 0, 0
        hit_rate, miss_rate = (self.hits / total) * 100, (self.misses / total) * 100
        return hit_rate, miss_rate, self.hits, self.misses

def to_signed32(value):
    value &= 0xFFFFFFFF; return value - (1 << 32) if value & (1 << 31) else value
def float_to_int_bits(f):
    return struct.unpack('I', struct.pack('f', float(f)))[0]
def int_bits_to_float(i):
    try: return struct.unpack('f', struct.pack('I', i & 0xFFFFFFFF))[0]
    except: return float('nan')
def format_instruction(instr):
    if not instr: return ""
    op = instr['op']
    if op in ["ADD", "SUB", "MUL", "DIV", "ADD.S", "MUL.S"]: return f"{op:<8} x{instr['rd']}, x{instr['rs1']}, x{instr['rs2']}"
    if op == "ADDI": return f"{op:<8} x{instr['rd']}, x{instr['rs1']}, {instr['imm']}"
    if op == "LW": return f"{op:<8} x{instr['rd']}, {instr['addr']}"
    if op == "SW": return f"{op:<8} x{instr['rs1']}, {instr['addr']}"
    if op == "BEQ": return f"{op:<8} x{instr['rs1']}, x{instr['rs2']}, target"
    if op == "J": return f"{op:<8} target"
    return str(instr)

# --- CORREÇÃO: Parser Unificado ---
def parse_program(asm_code):
    initial_lines = [line.strip().split('#')[0].strip() for line in asm_code.strip().splitlines()]
    lines = [line for line in initial_lines if line]
    labels, program_lines, current_addr = {}, [], 0
    for line in lines:
        if ':' in line:
            label, rest = line.split(':', 1); labels[label.strip()] = current_addr
            if rest.strip(): program_lines.append(rest.strip()); current_addr += 1
        else: program_lines.append(line); current_addr += 1
    program = []
    for addr, line in enumerate(program_lines):
        parts = line.replace(",", "").split()
        op = parts[0].upper()
        if op in ["ADD", "SUB", "MUL", "DIV", "ADD.S", "MUL.S"]: instr = {"op": op, "rd": int(parts[1][1:]), "rs1": int(parts[2][1:]), "rs2": int(parts[3][1:])}
        elif op == "ADDI": instr = {"op": op, "rd": int(parts[1][1:]), "rs1": int(parts[2][1:]), "imm": int(parts[3])}
        elif op == "LW": instr = {"op": op, "rd": int(parts[1][1:]), "addr": int(parts[2])}
        elif op == "SW": instr = {"op": op, "rs1": int(parts[1][1:]), "addr": int(parts[2])}
        elif op == "BEQ": instr = {"op": op, "rs1": int(parts[1][1:]), "rs2": int(parts[2][1:]), "target": labels[parts[3]]}
        elif op == "J": instr = {"op": op, "target": labels[parts[1]]}
        else: raise ValueError(f"Instrução desconhecida: {line}")
        program.append(instr)
    return program

# --- CORREÇÃO: Executor de Ciclo Único Atualizado ---
def execute_instruction_single_cycle(instr, regs, mem, pc):
    op = instr["op"]; INT32_MIN, INT32_MAX = -2**31, 2**31 - 1
    next_pc = pc + 1
    if op in ["ADD", "ADDI", "SUB", "MUL", "DIV"]:
        rs1_val = to_signed32(regs.read(instr["rs1"])); rs2_val = instr["imm"] if op == "ADDI" else to_signed32(regs.read(instr["rs2"]))
        if op in ["ADD", "ADDI"]: result = rs1_val + rs2_val
        elif op == "SUB": result = rs1_val - rs2_val
        elif op == "MUL": result = rs1_val * rs2_val
        elif op == "DIV":
            if rs2_val == 0: raise ValueError("Divisão por zero")
            if rs1_val == INT32_MIN and rs2_val == -1: raise OverflowError("Overflow na divisão")
            result = int(rs1_val / rs2_val)
        if not (INT32_MIN <= result <= INT32_MAX) and op != "DIV": raise OverflowError(f"Overflow na operação")
        regs.write(instr["rd"], result)
    elif op in ["ADD.S", "MUL.S"]:
        float1, float2 = int_bits_to_float(regs.read(instr["rs1"])), int_bits_to_float(regs.read(instr["rs2"]))
        result_float = (float1 + float2) if op == "ADD.S" else (float1 * float2)
        regs.write(instr["rd"], float_to_int_bits(result_float))
    elif op == "LW": regs.write(instr["rd"], mem.load_word(instr["addr"]))
    elif op == "SW": mem.store_word(instr["addr"], regs.read(instr["rs1"]))
    elif op == "BEQ":
        if regs.read(instr['rs1']) == regs.read(instr['rs2']):
            next_pc = instr['target']
    elif op == "J":
        next_pc = instr['target']
    else: raise ValueError(f"Instrução desconhecida {op}")
    return next_pc

def run_pipeline_simulation(program, regs, mem, cache, hazard_mode, cache_mode, miss_penalty):
    pc, cycle, stalls, mem_stalls = 0, 0, 0, 0
    pipeline_stages = {'IF': None, 'ID': None, 'EX': None, 'MEM_WB': None}
    history, hazard_logs = [], []
    while pc < len(program) or any(s is not None for s in pipeline_stages.values()):
        cycle += 1
        if mem_stalls > 0:
            mem_stalls -= 1; hazard_logs.append(f"Ciclo {cycle}: 🟡 STALL DE MEMÓRIA (Cache Miss)")
            history.append({ 'Ciclo': cycle, 'IF': "STALL", 'ID': "STALL", 'EX': "STALL", 'MEM_WB': format_instruction(pipeline_stages['MEM_WB'])})
            continue
        prev_stages = deepcopy(pipeline_stages); stall_this_cycle = False
        instr_wb = prev_stages['MEM_WB']
        if instr_wb:
            op = instr_wb['op']
            if op == "LW":
                data, status = cache.read(instr_wb['addr']) if cache_mode == 'enabled' else (mem.load_word(instr_wb['addr']), 'hit')
                if status == 'miss' and miss_penalty > 1: mem_stalls = miss_penalty - 1
                regs.write(instr_wb['rd'], data)
            elif op == "SW":
                _, status = cache.write(instr_wb['addr'], regs.read(instr_wb['rs1'])) if cache_mode == 'enabled' else (mem.store_word(instr_wb['addr'], regs.read(instr_wb['rs1'])), 'hit')
                if status == 'miss' and miss_penalty > 1: mem_stalls = miss_penalty - 1
            elif 'result' in instr_wb: regs.write(instr_wb['rd'], instr_wb['result'])
        instr_ex = prev_stages['EX']
        flush_if_id = False
        if instr_ex:
            op = instr_ex['op']
            if op == 'BEQ' and regs.read(instr_ex['rs1']) == regs.read(instr_ex['rs2']):
                pc = instr_ex['target']; flush_if_id = True; hazard_logs.append(f"Ciclo {cycle}: 🔴 CONTROL HAZARD (BEQ). Desvio tomado, flush IF/ID.")
            elif op == 'J':
                pc = instr_ex['target']; flush_if_id = True; hazard_logs.append(f"Ciclo {cycle}: 🔴 CONTROL HAZARD (J). Desvio tomado, flush IF/ID.")
            else:
                val1, val2 = regs.read(instr_ex.get('rs1',0)), regs.read(instr_ex.get('rs2',0)) if 'rs2' in instr_ex else instr_ex.get('imm',0)
                if hazard_mode == 'forwarding':
                    instr_mem = prev_stages['MEM_WB']
                    if instr_mem and 'result' in instr_mem and 'rd' in instr_mem:
                         fwd_reg = instr_mem['rd']
                         if fwd_reg == instr_ex.get('rs1'): val1 = instr_mem['result']; hazard_logs.append(f"Ciclo {cycle}: 🟢 FWD(MEM->EX) x{fwd_reg}")
                         if 'rs2' in instr_ex and fwd_reg == instr_ex.get('rs2'): val2 = instr_mem['result']; hazard_logs.append(f"Ciclo {cycle}: 🟢 FWD(MEM->EX) x{fwd_reg}")
                if op in ["ADD", "ADDI"]: instr_ex['result'] = to_signed32(val1) + to_signed32(val2)
                elif op == "SUB": instr_ex['result'] = to_signed32(val1) - to_signed32(val2)
            pipeline_stages['MEM_WB'] = instr_ex
        else: pipeline_stages['MEM_WB'] = None
        instr_id = prev_stages['ID']
        if instr_id:
            src_regs = {instr_id.get('rs1'), instr_id.get('rs2')} - {None}
            instr_ex_prev = prev_stages['EX']
            if instr_ex_prev and 'rd' in instr_ex_prev and instr_ex_prev['rd'] in src_regs:
                hazard_reg = instr_ex_prev['rd']
                if instr_ex_prev['op'] == 'LW':
                    if hazard_mode != 'none': stall_this_cycle = True; hazard_logs.append(f"Ciclo {cycle}: 🔴 HAZARD DE LOAD-USE x{hazard_reg}. Ação: STALL.")
                    else: hazard_logs.append(f"Ciclo {cycle}: 🔴 HAZARD DE LOAD-USE x{hazard_reg}. 🟡 Ação: Nenhuma.")
                else:
                    if hazard_mode == 'stall': stall_this_cycle = True; hazard_logs.append(f"Ciclo {cycle}: 🟡 HAZARD DE DADOS x{hazard_reg}. Ação: STALL.")
                    elif hazard_mode == 'none': hazard_logs.append(f"Ciclo {cycle}: 🔴 HAZARD DE DADOS x{hazard_reg}. 🟡 Ação: Nenhuma.")
        if stall_this_cycle: stalls += 1; pipeline_stages['EX'] = None
        elif flush_if_id: pipeline_stages['EX'] = prev_stages['ID']; pipeline_stages['ID'] = pipeline_stages['IF'] = None
        else:
            pipeline_stages['EX'] = prev_stages['ID']; pipeline_stages['ID'] = prev_stages['IF']
            if pc < len(program): pipeline_stages['IF'] = program[pc]; pc += 1
            else: pipeline_stages['IF'] = None
        history.append({ 'Ciclo': cycle, 'IF': format_instruction(pipeline_stages['IF']), 'ID': format_instruction(pipeline_stages['ID']), 'EX': f"*{format_instruction(pipeline_stages['EX'])}" if stall_this_cycle else format_instruction(pipeline_stages['EX']), 'MEM_WB': format_instruction(pipeline_stages['MEM_WB'])})
        if cycle > 200: hazard_logs.append("Simulação parada: limite de ciclos atingido."); break
    return history, hazard_logs, cycle, stalls

# ===================================================================
# PARTE B: LÓGICA DA INTERFACE STREAMLIT
# ===================================================================

def style_regs(df_row):
    prev_regs_list = st.session_state.get('prev_regs', [0]*32)
    reg_num_str = df_row.name
    if isinstance(reg_num_str, str) and reg_num_str.startswith('x'):
        reg_num = int(reg_num_str[1:])
        current_val = df_row['Valor Int']
        prev_val = to_signed32(prev_regs_list[reg_num])
        if current_val != prev_val:
            return ['background-color: #1E4620'] * len(df_row)
    return [''] * len(df_row)

def display_active_datapath(stages):
    st.markdown("##### Atividade dos Componentes por Estágio:")
    for stage_name, instr in reversed(list(stages.items())):
        if instr:
            op = instr['op']; activity = ""
            if stage_name == 'ID': activity = "🔹 **Banco de Registradores (Leitura)**"
            elif stage_name == 'EX':
                if op in ['ADD', 'ADDI', 'SUB', 'MUL', 'DIV', 'BEQ']: activity = "📐 **ALU (Cálculo Aritmético/Lógico)**"
                elif op in ['LW', 'SW']: activity = "📐 **ALU (Cálculo de Endereço)**"
            elif stage_name == 'MEM_WB':
                if op in ['LW', 'SW']: activity = "💾 **Memória/Cache (Acesso)**"
                if op not in ['SW', 'BEQ', 'J']: activity += " e ✍️ **Banco de Registradores (Escrita)**" if activity else "✍️ **Banco de Registradores (Escrita)**"
            st.markdown(f"**{stage_name}:** `{format_instruction(instr)}` -> {activity if activity else 'Avançando...'}")

def main():
    st.set_page_config(layout="wide", page_title="Simulador RISC-V")
    st.title("Simulador de Processador RISC-V")
    st.sidebar.header("Configurações da Simulação")
    selected_part = st.sidebar.radio("Modo de Simulação:", ('p1_2', 'p_extra'), index=1, format_func=lambda x: "Ciclo Único" if x == 'p1_2' else "Pipeline (Completo)")
    if selected_part == 'p_extra':
        hazard_mode = st.sidebar.radio("Tratamento de Hazard:", ('none', 'stall', 'forwarding'), index=2, format_func=lambda x: {'none':'Sem tratamento', 'stall':'Stall', 'forwarding':'Forwarding'}[x])
        cache_mode = st.sidebar.radio("Hierarquia de Memória:", ('enabled', 'disabled'), format_func=lambda x: "Com Cache" if x == 'enabled' else "Sem Cache")
        if cache_mode == 'enabled':
            cache_policy = st.sidebar.radio("Política de Cache:", ('lru', 'random'), format_func=str.upper)
            miss_penalty = st.sidebar.number_input("Penalidade de Miss (ciclos):", min_value=1, value=10)
        else: cache_policy, miss_penalty = 'lru', 1
    else: hazard_mode, cache_mode, cache_policy, miss_penalty = 'forwarding', 'disabled', 'lru', 1

    if 'program_loaded' not in st.session_state:
        st.session_state.program_loaded = False
        st.session_state.regs = RegisterFile(); st.session_state.mem = SimpleMemory(64)
        st.session_state.cache = Cache(num_sets=8, main_memory=st.session_state.mem)
        st.session_state.program, st.session_state.pc, st.session_state.p_pc, st.session_state.p_cycle = [], 0, 0, 0
        st.session_state.log, st.session_state.history, st.session_state.hazard_logs = [], [], []
        st.session_state.final_stats, st.session_state.prev_regs = {}, [0]*32
        st.session_state.p_pipeline_stages = {'IF': None, 'ID': None, 'EX': None, 'MEM_WB': None}
        st.session_state.p_stalls, st.session_state.p_mem_stalls = 0, 0

    st.subheader("Código Assembly")
    code = st.text_area("Escreva seu código aqui:", height=300, label_visibility="collapsed", key="code_area")
    cols = st.columns([1.5, 1.5, 1.5, 5])
    
    def reset_and_load():
        current_code = st.session_state.code_area
        st.session_state.log, st.session_state.history, st.session_state.hazard_logs, st.session_state.final_stats = [], [], [], {}
        st.session_state.program_loaded = False
        try:
            st.session_state.regs, st.session_state.mem = RegisterFile(), SimpleMemory(64)
            st.session_state.cache = Cache(num_sets=8, main_memory=st.session_state.mem, replacement_policy=cache_policy)
            st.session_state.program = parse_program(current_code) # Usa o parser unificado
            st.session_state.pc, st.session_state.p_pc, st.session_state.p_cycle = 0, 0, 0
            st.session_state.p_stalls, st.session_state.p_mem_stalls = 0, 0
            st.session_state.p_pipeline_stages = {'IF': None, 'ID': None, 'EX': None, 'MEM_WB': None}
            st.session_state.prev_regs = [0]*32
            st.session_state.log.append(f"✅ Programa carregado com {len(st.session_state.program)} instruções. Pronto para iniciar.")
            st.session_state.program_loaded = True
        except (ValueError, KeyError) as e:
            st.session_state.log.append(f"❌ ERRO DE SINTAXE: {e}"); st.session_state.program = []
        return st.session_state.program_loaded

    if cols[0].button("Carregar/Resetar", use_container_width=True):
        reset_and_load()

    step_button_label = "Próximo Passo ➡️" if selected_part == 'p1_2' else "Próximo Ciclo ➡️"
    if cols[1].button(step_button_label, use_container_width=True):
        if not st.session_state.get('program_loaded', False):
            if not reset_and_load(): return
        if selected_part == 'p1_2': execute_one_single_cycle_step()
        else: execute_one_pipeline_cycle(hazard_mode, cache_mode, miss_penalty)

    if cols[2].button("Executar Tudo ⏩", use_container_width=True):
        if reset_and_load():
            run_all(selected_part, hazard_mode, cache_mode, miss_penalty)

    st.subheader("Resultados da Simulação")
    if selected_part == 'p_extra':
        if st.session_state.hazard_logs:
            with st.expander("Detalhes de Hazards e Stalls"):
                for entry in st.session_state.hazard_logs: st.text(entry)
        if st.session_state.history:
            st.write("Tabela de Execução do Pipeline"); st.dataframe(pd.DataFrame(st.session_state.history).set_index('Ciclo').fillna(""), use_container_width=True)
    else:
        with st.expander("Log de Execução (Ciclo Único)", expanded=True):
            for entry in st.session_state.log: st.text(entry)
    
    if selected_part == 'p_extra' and st.session_state.get('p_pipeline_stages'):
        with st.expander("Datapath Ativo no Último Ciclo", expanded=True):
            display_active_datapath(st.session_state.p_pipeline_stages)
            
    st.subheader("Estado do Processador")
    tab_stats, tab_regs, tab_mem, tab_cache_stats, tab_cache_content = st.tabs(["📊 Estatísticas Gerais", "📟 Registradores", "💾 Memória", "⚡ Estatísticas da Cache", "🔬 Conteúdo da Cache"])
    with tab_stats:
        if st.session_state.final_stats:
            stats = st.session_state.final_stats; num_instr = stats.get('num_instr', 0); cycle = stats.get('cycle', 0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Número de Instruções", num_instr)
            c2.metric("Total de Ciclos", cycle)
            if selected_part == 'p_extra':
                cpi = cycle / num_instr if num_instr > 0 else 0
                c3.metric("CPI (Ciclos por Instrução)", f"{cpi:.2f}")
                speedup = num_instr / cycle if cycle > 0 else 0
                st.metric("Speedup (vs Ciclo Único)", f"{speedup:.2f}x")
        else: st.info("Execute um programa para ver as estatísticas de desempenho.")
    with tab_regs:
        if st.session_state.get('regs'):
            reg_data = [{"Reg": f"x{i}", "Valor Int": to_signed32(st.session_state.regs.read(i))} for i in range(32)]
            df_regs = pd.DataFrame(reg_data).set_index('Reg')
            st.dataframe(df_regs.style.apply(style_regs, axis=1), use_container_width=True, height=400)
    with tab_mem:
        if st.session_state.get('mem'):
            mem_data = [{"Endereço": i, "Valor": st.session_state.mem.load_word(i)} for i in range(st.session_state.mem.size)]
            st.dataframe(pd.DataFrame(mem_data), use_container_width=True, height=400)
    with tab_cache_stats:
        if st.session_state.get('cache') and cache_mode == 'enabled':
            hit_rate, miss_rate, hits, misses = st.session_state.cache.get_stats()
            c1, c2 = st.columns(2); c1.metric("Taxa de Acertos (Hit Rate)", f"{hit_rate:.2f}%"); c2.metric("Taxa de Falhas (Miss Rate)", f"{miss_rate:.2f}%")
            c1, c2 = st.columns(2); c1.metric("Total de Hits", hits); c2.metric("Total de Misses", misses)
        else: st.info("A cache está desabilitada.")
    with tab_cache_content:
        if st.session_state.get('cache') and cache_mode == 'enabled':
            cache_content = []
            for i, s in enumerate(st.session_state.cache.sets):
                for j, line in enumerate(s):
                    cache_content.append({ "Conjunto": i, "Via": j, "Válido": line['valid'], "LRU": line['lru'], "Tag": line['tag'], "Dado": to_signed32(line['data']) })
            st.dataframe(pd.DataFrame(cache_content), use_container_width=True)
        else: st.info("A cache está desabilitada.")

def execute_one_single_cycle_step():
    ss = st.session_state
    if not ss.get('program_loaded', False):
        ss.log.append("🟡 Por favor, carregue um programa primeiro clicando em 'Carregar/Resetar'."); return
    ss.prev_regs = [ss.regs.read(i) for i in range(32)]
    if not (0 <= ss.pc < len(ss.program)):
        if 'ended' not in ss or not ss['ended']: ss.log.append("🏁 Fim do programa."); ss['ended'] = True
        return
    instr = ss.program[ss.pc]
    try:
        ss.pc = execute_instruction_single_cycle(instr, ss.regs, ss.mem, ss.pc)
        ss.log.append(f"🟢 Passo {ss.pc}: {format_instruction(instr)} -> Sucesso!")
    except (ValueError, IndexError, OverflowError) as e:
        ss.log.append(f"🔴 Passo {ss.pc+1}: {format_instruction(instr)} -> ERRO: {e}")
        ss.pc += 1
    ss.final_stats = {'cycle': len(ss.log) - 1, 'stalls': 0, 'num_instr': len(ss.program)}; ss['ended'] = False

def execute_one_pipeline_cycle(hazard_mode, cache_mode, miss_penalty):
    ss = st.session_state
    if not ss.get('program_loaded', False):
        ss.log.append("🟡 Por favor, carregue um programa primeiro clicando em 'Carregar/Resetar'."); return
    ss.prev_regs = [ss.regs.read(i) for i in range(32)]
    if ss.p_pc >= len(ss.program) and all(s is None for s in ss.p_pipeline_stages.values()):
        if 'ended' not in ss or not ss['ended']: ss.log.append("🏁 Fim do programa."); ss['ended'] = True
        return
    
    ss.p_cycle += 1; stall_this_cycle = False
    if ss.p_mem_stalls > 0:
        ss.p_mem_stalls -= 1; ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🟡 STALL DE MEMÓRIA (Cache Miss)")
        ss.history.append({ 'Ciclo': ss.p_cycle, 'IF': "STALL", 'ID': "STALL", 'EX': "STALL", 'MEM_WB': format_instruction(ss.p_pipeline_stages['MEM_WB'])})
        return

    prev_stages = deepcopy(ss.p_pipeline_stages)
    instr_wb = prev_stages['MEM_WB']
    if instr_wb:
        op = instr_wb['op']
        if op == "LW":
            data, status = ss.cache.read(instr_wb['addr']) if cache_mode == 'enabled' else (ss.mem.load_word(instr_wb['addr']), 'hit')
            if status == 'miss' and miss_penalty > 1: ss.p_mem_stalls = miss_penalty - 1
            ss.regs.write(instr_wb['rd'], data)
        elif op == "SW":
            _, status = ss.cache.write(instr_wb['addr'], ss.regs.read(instr_wb['rs1'])) if cache_mode == 'enabled' else (ss.mem.store_word(instr_wb['addr'], ss.regs.read(instr_wb['rs1'])), 'hit')
            if status == 'miss' and miss_penalty > 1: ss.p_mem_stalls = miss_penalty - 1
        elif 'result' in instr_wb: ss.regs.write(instr_wb['rd'], instr_wb['result'])
    instr_ex = prev_stages['EX']; flush_if_id = False
    if instr_ex:
        op = instr_ex['op']
        if op == 'BEQ' and ss.regs.read(instr_ex['rs1']) == ss.regs.read(instr_ex['rs2']):
            ss.p_pc = instr_ex['target']; flush_if_id = True; ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🔴 CONTROL HAZARD (BEQ). Desvio tomado, flush IF/ID.")
        elif op == 'J':
            ss.p_pc = instr_ex['target']; flush_if_id = True; ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🔴 CONTROL HAZARD (J). Desvio tomado, flush IF/ID.")
        else:
            val1, val2 = ss.regs.read(instr_ex.get('rs1',0)), ss.regs.read(instr_ex.get('rs2',0)) if 'rs2' in instr_ex else instr_ex.get('imm',0)
            if hazard_mode == 'forwarding':
                instr_mem = prev_stages['MEM_WB']
                if instr_mem and 'result' in instr_mem and 'rd' in instr_mem:
                     fwd_reg = instr_mem['rd']
                     if fwd_reg == instr_ex.get('rs1'): val1 = instr_mem['result']; ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🟢 FWD(MEM->EX) x{fwd_reg}")
                     if 'rs2' in instr_ex and fwd_reg == instr_ex.get('rs2'): val2 = instr_mem['result']; ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🟢 FWD(MEM->EX) x{fwd_reg}")
            if op in ["ADD", "ADDI"]: instr_ex['result'] = to_signed32(val1) + to_signed32(val2)
            elif op == "SUB": instr_ex['result'] = to_signed32(val1) - to_signed32(val2)
        ss.p_pipeline_stages['MEM_WB'] = instr_ex
    else: ss.p_pipeline_stages['MEM_WB'] = None
    instr_id = prev_stages['ID']
    if instr_id:
        src_regs = {instr_id.get('rs1'), instr_id.get('rs2')} - {None}
        instr_ex_prev = prev_stages['EX']
        if instr_ex_prev and 'rd' in instr_ex_prev and instr_ex_prev['rd'] in src_regs:
            hazard_reg = instr_ex_prev['rd']
            if instr_ex_prev['op'] == 'LW':
                if hazard_mode != 'none': stall_this_cycle = True; ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🔴 HAZARD DE LOAD-USE x{hazard_reg}. Ação: STALL.")
                else: ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🔴 HAZARD DE LOAD-USE x{hazard_reg}. 🟡 Ação: Nenhuma.")
            else:
                if hazard_mode == 'stall': stall_this_cycle = True; ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🟡 HAZARD DE DADOS x{hazard_reg}. Ação: STALL.")
                elif hazard_mode == 'none': ss.hazard_logs.append(f"Ciclo {ss.p_cycle}: 🔴 HAZARD DE DADOS x{hazard_reg}. 🟡 Ação: Nenhuma.")
    if stall_this_cycle: ss.p_stalls += 1; ss.p_pipeline_stages['EX'] = None
    elif flush_if_id: ss.p_pipeline_stages['EX'] = prev_stages['ID']; ss.p_pipeline_stages['ID'] = ss.p_pipeline_stages['IF'] = None
    else:
        ss.p_pipeline_stages['EX'] = prev_stages['ID']; ss.p_pipeline_stages['ID'] = prev_stages['IF']
        if ss.p_pc < len(ss.program): ss.p_pipeline_stages['IF'] = ss.program[ss.p_pc]; ss.p_pc += 1
        else: ss.p_pipeline_stages['IF'] = None
    ss.history.append({ 'Ciclo': ss.p_cycle, 'IF': format_instruction(ss.p_pipeline_stages['IF']), 'ID': format_instruction(ss.p_pipeline_stages['ID']), 'EX': f"*{format_instruction(ss.p_pipeline_stages['EX'])}" if stall_this_cycle else format_instruction(ss.p_pipeline_stages['EX']), 'MEM_WB': format_instruction(ss.p_pipeline_stages['MEM_WB'])})
    ss.final_stats = {'cycle': ss.p_cycle, 'stalls': ss.p_stalls, 'num_instr': len(ss.program)}; ss['ended'] = False

def run_all(selected_part, hazard_mode, cache_mode, miss_penalty):
    if selected_part == 'p1_2':
        st.session_state.pc = 0
        if not st.session_state.program: return
        while st.session_state.pc < len(st.session_state.program): execute_one_single_cycle_step()
    else:
        while True:
            is_done = st.session_state.p_pc >= len(st.session_state.program) and all(s is None for s in st.session_state.p_pipeline_stages.values())
            if is_done or st.session_state.p_cycle > 200: break
            execute_one_pipeline_cycle(hazard_mode, cache_mode, miss_penalty)
    st.session_state.log.append("🏁 Fim do programa.")

if __name__ == "__main__":
    main()