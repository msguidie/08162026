# Cross-language validation: Node oracle -> JSONL -> Python checker.
#
#   make all                       # full gate: 2000 games x 6 configs, then check
#   make GAMES=200 all             # quick pass
#   make generate / make check     # the two halves separately
#   make stress                    # noble/orphan + full-chaos coverage batches
#   make bench                     # Python engine throughput only
#   make clean

PY     ?= /home/user/venv-splendor/bin/python
NODE   ?= node
GAMES  ?= 2000
SEED   ?= 1
OUT    ?= data
GEN    := $(NODE) gen_trajectories.js
CHECK  := $(PY) replay_check.py

# mode / seats / layout / seed offset — one file per configuration
CONFIGS := ind2 ind3 ind4 ovt team_adj team_opp

.PHONY: all generate check stress bench clean

all: generate check

generate: $(OUT)
	$(GEN) --out $(OUT)/ind2.jsonl.gz     --games $(GAMES) --mode INDIVIDUAL --players 2 --seed $$(($(SEED)+1))
	$(GEN) --out $(OUT)/ind3.jsonl.gz     --games $(GAMES) --mode INDIVIDUAL --players 3 --seed $$(($(SEED)+2))
	$(GEN) --out $(OUT)/ind4.jsonl.gz     --games $(GAMES) --mode INDIVIDUAL --players 4 --seed $$(($(SEED)+3))
	$(GEN) --out $(OUT)/ovt.jsonl.gz      --games $(GAMES) --mode ONE_V_TWO              --seed $$(($(SEED)+4))
	$(GEN) --out $(OUT)/team_adj.jsonl.gz --games $(GAMES) --mode TEAM --layout ADJACENT --seed $$(($(SEED)+5))
	$(GEN) --out $(OUT)/team_opp.jsonl.gz --games $(GAMES) --mode TEAM --layout OPPOSITE --seed $$(($(SEED)+6))

# Extra coverage: heavy tableau growth (three-noble and orphaned-choice states)
# and an all-chaos batch (resign / timeout on almost every game).
stress: $(OUT)
	$(GEN) --out $(OUT)/stress_nobles.jsonl.gz --games $(GAMES) --mode INDIVIDUAL --players 4 \
		--seed $$(($(SEED)+11)) --t1-bias 0.85 --chaos-frac 0 --orphan-hunt --perm-check-p 0.002
	$(GEN) --out $(OUT)/stress_nobles3.jsonl.gz --games $(GAMES) --mode ONE_V_TWO \
		--seed $$(($(SEED)+12)) --t1-bias 0.85 --chaos-frac 0 --orphan-hunt --perm-check-p 0.002
	$(GEN) --out $(OUT)/stress_chaos.jsonl.gz --games $(GAMES) --mode INDIVIDUAL --players 4 \
		--seed $$(($(SEED)+13)) --chaos-frac 1.0 --resign-p 0.03 --timeout-p 0.02
	$(GEN) --out $(OUT)/stress_chaos_team.jsonl.gz --games $(GAMES) --mode TEAM --layout OPPOSITE \
		--seed $$(($(SEED)+14)) --chaos-frac 1.0 --resign-p 0.03 --timeout-p 0.02

check:
	$(CHECK) $(OUT) --bench 200000

bench:
	$(CHECK) --bench 300000

$(OUT):
	mkdir -p $(OUT)

clean:
	rm -rf $(OUT)
