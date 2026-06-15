"""Ecosystem-role tiers for Ethereum L1 repos — encodes structural function as a prior."""

ROLE_MAP = {
    # Execution clients — run the EVM, hold state
    'execution_client': ['go-ethereum','nethermind','besu','erigon','reth','ethrex','silkworm','evmone'],
    # Consensus clients — drive proof-of-stake
    'consensus_client': ['lighthouse','prysm','teku','nimbus-eth2','lodestar','grandine','lambda_ethereum_consensus'],
    # Core specs / standards — define the protocol
    'core_spec': ['consensus-specs','eips','execution-apis','execution-specs'],
    # Languages & compilers
    'language': ['solidity','vyper','fe'],
    # Dev frameworks
    'framework': ['hardhat','foundry','remix-project','ape','scaffold-eth-2','titanoboa','tevm-monorepo','hardhat-deploy'],
    # Smart-contract libraries / SDKs
    'library': ['openzeppelin-contracts','ethers.js','web3.py','web3j','viem','alloy','nethereum',
                'solady','account-abstraction','safe-smart-account','solidity-lib','stylus-sdk-rs','whatsabi','libp2p'],
    # Cryptographic primitives
    'crypto': ['blst','gnark-crypto','py_ecc','mcl','libbls','bls','algebra','lambdaworks',
               'snark-verifier','jellyfish','noble-curves','js-ethereum-cryptography'],
    # ZK / proving systems
    'zk': ['risc0-ethereum','sp1','op-succinct','plonky3','powdr','miden-vm','rsp','halmos'],
    # MEV / block-building infra
    'mev': ['mev-boost','mev-boost-relay','rbuilder','commit-boost-client'],
    # Node infra / devops
    'infra': ['eth-docker','dappnode','checkpointz','ethereum-helm-charts','ethereum-package',
              'simple-optimism-node','ethstaker-deposit-cli','ethdo'],
    # Analytics / explorers / data
    'analytics': ['l2beat','defillama-adapters','chainlist','chains','blockscout','otterscan',
                  'trueblocks-core'],
    # Security / verification / audit tooling
    'security': ['aderyn','certoraprover','hevm','act','solhint','sourcify','goevmlab',
                 'edb','ethdebug','intellij-solidity','swiss-knife','format'],
    # L2 / scaling
    'l2': ['taiko-mono','helios'],
    # Misc tooling / infra graph
    'tooling': ['dependency-graph'],
}

# Prior importance level per role (1=peripheral .. 5=foundational), a single numeric feature.
TIER_PRIOR = {
    'core_spec': 5.0, 'execution_client': 5.0, 'consensus_client': 4.7, 'language': 4.5,
    'library': 3.8, 'framework': 3.7, 'crypto': 3.3, 'mev': 3.2, 'zk': 3.0,
    'analytics': 2.8, 'security': 2.6, 'l2': 2.6, 'infra': 2.5, 'tooling': 2.3, 'other': 2.0,
}

def repo_role(slug):
    name = slug.split('/')[-1].lower()
    for role, names in ROLE_MAP.items():
        if name in names:
            return role
    return 'other'

def role_features(slug):
    role = repo_role(slug)
    return {'tier_prior': TIER_PRIOR.get(role, 2.0), 'role': role}
