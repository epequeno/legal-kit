#!/usr/bin/env python3
"""Generate detailed reasoning chains for batch 12 legal entailment triples."""

import json

def generate_reasoning(idx, premise, hypothesis, label):
    """Generate legal reasoning for each specific case."""
    
    reasonings = {
        0: """The passage discusses securities filing obligations and states parties must "cooperate in good faith to seek confidential treatment" for agreement terms included in regulatory filings. The hypothesis claims parties must cooperate in good faith to obtain confidential treatment for agreement terms in securities filings. This represents direct entailment - the passage explicitly establishes this cooperation duty regarding confidential treatment of contractual terms when they appear in SEC or similar regulatory disclosures. The legal structure involves mandatory cooperation between contracting parties to protect sensitive business terms from public disclosure through confidential treatment requests. Both the cooperation obligation and its specific application to securities filings containing agreement terms are textually confirmed.""",
        
        1: """The passage establishes a Mutual Non-Disclosure Agreement that defines "Confidential Information" and sets forth protections requiring the receiving party to treat disclosed information as confidential. The hypothesis claims the agreement allows the receiving party to disclose confidential information to third parties without the disclosing party's consent. This represents clear contradiction - the foundational purpose of an NDA is to restrict unauthorized disclosure and require consent for sharing protected information. The passage's definitional framework establishing confidentiality protections and the agreement's very title indicating non-disclosure obligations are fundamentally incompatible with permitting free third-party disclosure without consent. The legal structure of mutual confidentiality prohibits the disclosure rights claimed.""",
        
        2: """The passage explicitly states in Item Four, Section 5 that "Payment method: Wire or transfer." The hypothesis claims the payment method for the recycling project is wire transfer or bank transfer. This represents clear entailment through direct textual correspondence. Wire transfer and bank transfer are functionally synonymous payment mechanisms in commercial transactions, representing electronic funds transfer methods. The passage identifies the precise payment mechanism that will be used for the energy saving service fees under the recycling project agreement. The legal structure specifies the permissible payment methods, and the hypothesis accurately captures these specified electronic payment options. Both wire and transfer refer to the same category of payment mechanisms.""",
        
        3: """The hypothesis discusses requirements related to "excess parachute payments" under Section 280G(b)(1) of the tax code, which governs change-in-control compensation arrangements. The passage addresses entirely different tax matters: Section 355 distributions (tax-free spinoffs) and participation in listed transactions under Treasury regulations. These represent distinct areas of tax law serving unrelated purposes - one involves executive compensation in change-of-control scenarios, the other involves corporate reorganizations and tax shelter transactions. The passage provides no information about parachute payments, excess compensation determinations, or Section 280G analysis. Without any substantive overlap between the tax provisions discussed, the logical relationship is neutral. The documents appear to address different transactional contexts.""",
        
        4: """The passage explicitly lists in numbered item 12 that services include "up to 10 mailboxes accessible via the POP3 mail protocol that are mapped to the Hosted Site's Internet address." The hypothesis claims the agreement provides the customer with up to 10 mailboxes accessible via the POP3 mail protocol. This constitutes clear entailment through direct textual correspondence. The passage specifies the exact number of mailboxes (10), the protocol (POP3), and confirms these are provided to the customer as part of the hosting services. The technical specification regarding email infrastructure is explicitly stated in the enumerated services section. The legal structure establishes definite service obligations with specific technical parameters.""",
        
        5: """The passage in Section 6.1 grants CERES the right to "take any action to enforce and to initiate and prosecute suits for infringement of jointly owned intellectual property rights covering JOINT INTELLECTUAL PROPERTY." It further states "CERES and IGER will consult with each other upon a course of action and enforcement strategy." The hypothesis claims CERES has the right to take enforcement action and prosecute infringement suits for jointly owned IP, in consultation with IGER. This represents clear entailment - the passage explicitly establishes both CERES's enforcement rights and the consultation requirement with IGER. The legal structure allocates primary enforcement authority to CERES while maintaining IGER's participatory rights through mandatory consultation. Both elements of the hypothesis are textually confirmed.""",
        
        6: """The hypothesis references "Conduct of Business" under Section 3.3.1, suggesting operational governance provisions. The passage presents detailed GaN equipment specifications with asset IDs, tool names, tool IDs, applications, and invoice tracking across multiple time periods. These represent entirely unrelated document sections - one would address business operations, management, and governance standards, while the other provides technical equipment inventory and financial tracking data. The passage contains no information about business conduct rules, operational requirements, or management obligations. Without any substantive connection between equipment specifications and business conduct provisions, the logical relationship is neutral. These appear to be different sections of a broader agreement.""",
        
        7: """The hypothesis makes a statement about Section 3.3.1 "Conduct of Business," which would typically address operational governance, management practices, or business conduct standards. The passage contains detailed GaN equipment tables listing six items with asset identifications, tool specifications, applications (all redacted as [***]), and invoice amounts tracked across multiple months from 2017-2019. These represent completely different document sections with no substantive overlap - technical equipment inventory versus business operations governance. The equipment data provides no information about conduct standards, operational requirements, or business management provisions. Without any connection between the subject matters, the relationship is neutral. The sections serve distinct documentary purposes.""",
        
        8: """The passage in Section (c) states that "Each party acknowledges that it is giving up the right to a trial by jury or court" and establishes that matters "shall have been submitted to arbitration as herein provided" with enforcement limited to confirming arbitration awards. The hypothesis claims the parties have agreed to resolve any disputes through binding arbitration rather than through a court trial. This represents clear entailment - the passage explicitly establishes the arbitration framework as the exclusive dispute resolution mechanism and documents the parties' waiver of jury trial and court litigation rights. The legal structure creates mandatory arbitration with judicial involvement limited to award confirmation, excluding traditional court trials as the primary dispute resolution forum.""",
        
        9: """The passage in Section 1 defines "Confidential Information" to mean "any confidential, proprietary, financial or otherwise non-public information, in written or electronic form, relating to the Disclosing Party, or the Purpose that is disclosed...by the Disclosing Party to the Recipient" and specifically requires it "must be expressly marked or designated in writing by the Disclosing Party in a manner to indicate its confidential, proprietary or otherwise non-public nature." The hypothesis accurately restates this definition including both the scope of protected information and the express written marking requirement. This represents clear entailment through direct textual correspondence. The passage establishes the definitional elements - information types, disclosure context, and marking requirement - that are accurately captured in the hypothesis.""",
        
        10: """The passage in Section 2 requires the Data Recipient to "report to DOHMH in writing any unauthorized or inadvertent use or disclosure" within specified timeframes. Section 3 explicitly mandates that "Data Recipient shall also assist in any subsequent investigation of the unauthorized or inadvertent use or disclosure and mitigate any possible resulting damages." The hypothesis claims the data recipient is NOT obligated to assist in any investigation of unauthorized or inadvertent data disclosure. This represents clear contradiction - Section 3 expressly imposes a duty to assist in investigations, directly opposing the claim that no such obligation exists. The legal structure creates affirmative cooperation duties that cannot be reconciled with the hypothesis's assertion of no assistance obligation.""",
        
        11: """The hypothesis discusses Diplomat's payment obligations to Tadeo under EXHIBIT A, while the passage only describes Tadeo's service provision obligations referencing EXHIBIT A. These address different directional obligations in the relationship. The passage doesn't contain the specific payment terms or conditions mentioned in the hypothesis, creating a neutral relationship.""",
        
        12: """The hypothesis describes WebMD's content provision obligations for anatomical drawings, while the passage establishes trade secret protection and confidentiality obligations. These are different contractual provisions - one about deliverables, the other about information protection. The passage contains no information about content delivery specifications, making this neutral.""",
        
        13: """The passage states "The relevant assignee Party shall have the sole responsibility, at its sole cost and expense, to file...forms or documents" to record IP transfers. The hypothesis claims the assignee must bear all costs for filing documents to record transfers. The passage explicitly allocates all filing costs to the assignee.""",
        
        14: """The hypothesis discusses Blue Hills Parties' representations about losses and material adverse effects from Section (g), while the passage addresses contribution and indemnification allocation from Section 4.2.2. These cover different contractual topics - representations/warranties versus liability allocation. The passage contains no information about loss representations, making this neutral.""",
        
        15: """The passage states in Section (a) "Operator shall be responsible, at its sole cost and expense, for maintaining and complying with all Permits required to perform the Facility Services." The hypothesis claims the operator must cover costs of obtaining and maintaining necessary permits. The passage explicitly assigns permit costs and compliance responsibilities to the Operator.""",
        
        16: """The passage requires signers to initial and attest to compliance with SSI standards "in accordance with 49 CFR Part 1520" and related guidance. The hypothesis claims the signer is NOT required to comply with SSI standards. This directly contradicts the attestation and compliance obligations in the SSI section, which requires compliance with specific regulatory standards.""",
        
        17: """Section 7.7 states the agreement "shall be construed as if drafted jointly by the Parties and no presumption or burden of proof shall arise favoring or disfavoring any Party by virtue of the authorship." The hypothesis claims one party drafted and imposed the agreement with drafting party's intent prevailing. This contradicts the joint construction principle and anti-presumption rule.""",
        
        18: """Section 7.3 requires the Indemnified Party to "(b) cooperate with the Indemnifying Party (at the Indemnifying Party's expense) in connection with the defense and settlement of the Claim." The hypothesis accurately restates this cooperation duty and expense allocation. The passage explicitly establishes both the cooperation obligation and the cost responsibility.""",
        
        19: """The passage allows CONSULTANT to reject unsuitable MANDATORY PRODUCT and requires ADAMS GOLF to supply acceptable product or terminate within 30 days. Section 8 then creates repayment obligations if minimum tournaments aren't met. The hypothesis claims no repayment if prevented from competing. The passage doesn't exempt prevented circumstances from repayment - it maintains the obligation.""",
        
        20: """The hypothesis describes a Strategic Alliance Agreement execution date and parties, while the passage defines "Tangible Net Worth" with specific accounting treatment of intangibles. These come from different document sections with no substantive connection. The passage contains no information about alliance agreement formation, making this neutral.""",
        
        21: """Section 2.A states franchisee's owners "shall not directly or indirectly...own, operate, lease franchise, engage in...any restaurant business" or similar businesses during the term. Section 2.B extends this post-term. The hypothesis accurately characterizes these comprehensive non-competition obligations during and after the franchise term.""",
        
        22: """The passage establishes a Confidentiality and Non-Disclosure Agreement with effective date and party identification. The hypothesis claims the agreement outlines intellectual property rights transfer terms. NDAs govern information protection, not IP transfer. The passage contains no provisions about transferring IP ownership or rights.""",
        
        23: """The table of contents lists sections including "DISCLOSURE OF CONFIDENTIAL INFORMATION," "RESTRICTIONS ON DISCLOSURE AND USE," and "STANDARD OF CARE." The hypothesis claims the agreement establishes confidentiality obligations between the parties. The structural outline and section titles clearly indicate the agreement's purpose is to establish confidential information handling duties.""",
        
        24: """The passage states "THIS NON-DISCLOSURE AGREEMENT...is made and entered into this _____ day of ___________, 2015 (the 'Effective Date')." The hypothesis claims the agreement was made in 2014. The passage explicitly specifies 2015 as the execution year, directly contradicting the 2014 claim.""",
        
        25: """The hypothesis presents an "INTELLECTUAL PROPERTY AGREEMENT" title and party identification, while the passage shows intellectual property certificates and consumer data contracts. These represent different document sections - one is a header, the other is an appendix or exhibit. Without substantive overlap in provisions, the relationship is neutral.""",
        
        26: """The hypothesis presents a Non-disclosure Agreement header identifying entrusting and recipient parties, while the passage defines confidential information and disclosure restrictions. These appear to be different sections of the same document type. The header identifies parties while the passage contains substantive provisions, creating a neutral relationship without direct logical connection.""",
        
        27: """Section 4.2.2 allows immediate termination "in the event of (a) a voluntary assignment by the other party for the benefit of creditors; (b) the institution of voluntary proceedings by the other party in bankruptcy." The hypothesis claims parties may NOT terminate for bankruptcy. This directly contradicts the express termination rights for bankruptcy and insolvency proceedings.""",
        
        28: """The passage describes IGER granting CERES exclusive licensing rights to jointly owned IP and collaboration results, with remuneration negotiated in good faith per Exhibit H. The hypothesis discusses CERES's enforcement rights for jointly owned IP from Section 6.1. These address different aspects - licensing/commercialization versus enforcement - from different sections.""",
        
        29: """The passage lists six GaN equipment items with asset IDs, tool names, and applications (all redacted as [***]). The hypothesis claims invoice amounts are provided for different time periods. Equipment listings and invoice amounts are related but distinct data categories. The passage shows equipment inventory, not the payment data referenced in the hypothesis.""",
        
        30: """The passage establishes Blue Hills Parties and Agent contribution obligations for losses and liability allocation based on fees and gross proceeds percentages. The hypothesis discusses representations about losses from calamities and material adverse effects. These cover different contractual functions - indemnification mechanics versus business condition representations - making them neutral.""",
        
        31: """The passage requires CONSULTANT to notify ADAMS GOLF if MANDATORY PRODUCT is unsuitable, giving ADAMS GOLF 30 days to supply acceptable product or terminate. Section 8 creates tournament participation minimums with repayment obligations. The hypothesis claims no repayment if prevented from competing beyond consultant's control. The passage doesn't provide this exception.""",
        
        32: """The passage addresses contribution allocation between Blue Hills Parties and Agent for losses and claims, based on percentage of fees to gross proceeds. The hypothesis discusses Blue Hills Parties' representations about not sustaining material losses from specified calamities. These cover different contractual provisions - liability allocation versus business representations - without direct connection.""",
        
        33: """The passage defines "Confidential Information" in Section 1 to include information "disclosed before, on or after the Effective Date" that is "expressly marked or designated in writing" as confidential. The hypothesis accurately restates this definition including the requirement for express written designation. The passage provides the specific definitional elements claimed.""",
        
        34: """Section 2 requires Data Recipient to "report to DOHMH in writing any unauthorized or inadvertent use or disclosure" within 3 business days. Section 3 requires Data Recipient to "assist in any subsequent investigation" of unauthorized disclosure. The hypothesis claims no obligation to assist investigations. This contradicts the express Section 3 duty to assist.""",
        
        35: """The passage states "Tadeo shall provide...the Online Hosting Services described on EXHIBIT A...at the cost specified and on the other terms and conditions as set forth on EXHIBIT A." The hypothesis discusses Diplomat's payment obligations to Tadeo per EXHIBIT A. While related financially, these describe different directional obligations without the passage confirming the specific payment terms.""",
        
        36: """Section (c) defines "Trade Secrets" and Section (d) requires return of Confidential Information at the disclosing party's request. The hypothesis describes WebMD's content provision obligations for anatomical drawings. These address different contractual subjects - information protection versus content deliverables - from apparently different agreement sections.""",
        
        37: """The passage states "The relevant assignee Party shall have the sole responsibility, at its sole cost and expense, to file the Intellectual Property Assignment Agreements" to record IP transfers. The hypothesis claims the assignee must bear all costs for filing necessary documents to record transfers. The passage explicitly assigns all filing costs to the assignee.""",
        
        38: """The passage addresses contribution allocation and relative fault determination between Blue Hills Parties and Agent for indemnification claims. The hypothesis presents representations about Blue Hills Parties not sustaining losses from calamities and material adverse effects. These cover different provisions - indemnification procedures versus business condition warranties.""",
        
        39: """Section (a) requires "Operator shall be responsible, at its sole cost and expense, for maintaining and complying with all Permits required to perform the Facility Services." The hypothesis claims the operator must cover costs of obtaining and maintaining necessary permits. The passage explicitly allocates permit costs and compliance obligations to Operator.""",
        
        40: """The passage requires signers to initial and attest to familiarity with and compliance with SSI requirements "in accordance with 49 CFR Part 1520" and supplementary DHS guidance. The hypothesis claims signers are NOT required to comply with SSI standards. This directly contradicts the attestation requirement for compliance with specified SSI regulatory standards.""",
        
        41: """Section 7.7 states "this Agreement shall be construed as if drafted jointly by the Parties and no presumption or burden of proof shall arise favoring or disfavoring any Party by virtue of the authorship of any of the provisions." The hypothesis claims one party drafted and imposed terms with that party's intent prevailing. This contradicts the express joint construction rule.""",
        
        42: """Section 7.3 requires the Indemnified Party to "(b) cooperate with the Indemnifying Party (at the Indemnifying Party's expense) in connection with the defense and settlement of the Claim." The hypothesis accurately captures this cooperation duty and expense allocation. The passage explicitly establishes both the cooperation obligation and cost responsibility structure.""",
        
        43: """The passage states CONSULTANT must achieve "satisfactory record of play in a minimum of [*****] professional golf association events" with repayment required for shortfalls. Section 8.C begins "In the event that CONSULTANT is prevented from competing for" but is truncated. The hypothesis claims no repayment if prevented from competing, but the passage is incomplete.""",
        
        44: """The passage shows "Strategic Alliance Agreement" as a title with execution date and parties "Freedom Mortgage Corporation" and "Cherry Hill Mortgage Investment Corp." The hypothesis defines "Tangible Net Worth" with detailed accounting treatments. These represent different document sections - header versus definitions - without substantive logical connection.""",
        
        45: """Section 2.A requires franchisee's owners not to "directly or indirectly...own, operate, lease franchise, engage in, be connected with...any restaurant business" or similar businesses during the franchise term. Section 2.B extends this post-term for two years. The hypothesis accurately describes these confidentiality and non-competition obligations during and after the term.""",
        
        46: """The passage establishes a "Confidentiality and Non-Disclosure Agreement" with effective date and party identification. The hypothesis claims the agreement outlines intellectual property rights transfer terms. NDAs govern information protection, not ownership transfer. The passage contains no provisions about transferring or assigning IP rights between parties.""",
        
        47: """The table of contents lists sections including "DISCLOSURE OF CONFIDENTIAL INFORMATION," "RESTRICTIONS ON DISCLOSURE AND USE," "STANDARD OF CARE," and "RETURN OF MATERIAL." The hypothesis claims the agreement establishes confidentiality obligations between AfriGIS and Client. The section structure clearly indicates the agreement's purpose is establishing confidential information handling obligations.""",
        
        48: """The passage states "THIS NON-DISCLOSURE AGREEMENT...is made and entered into this _____ day of ___________, 2015 (the 'Effective Date')." The hypothesis claims the agreement was made and entered into in 2014. The passage explicitly specifies 2015 as the year of execution, directly contradicting the 2014 claim.""",
        
        49: """Section 4.2.2 allows either party to "terminate this Agreement immediately by giving the other party written notice thereof in the event such other party...the institution of voluntary proceedings by the other party in bankruptcy, insolvency, moratorium or for a receivership." The hypothesis claims parties may NOT terminate for bankruptcy. This contradicts the express termination right."""
    }
    
    return reasonings.get(idx, f"Reasoning for case {idx}")

def main():
    # Read input
    with open('/Users/steven/code/legal-kit/data/sft_reasoning/batches/batch_12.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} triples\n")
    
    # Process each triple
    results = []
    for idx, item in enumerate(data):
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = item['label']
        
        # Generate reasoning
        reasoning = generate_reasoning(idx, premise, hypothesis, label)
        
        # Count words
        word_count = len(reasoning.split())
        
        result = {
            'input': f"Passage: {premise}\n\nClaim: {hypothesis}",
            'output': label,
            'label': label,
            'reasoning': reasoning,
            'reasoning_word_count': word_count
        }
        
        results.append(result)
        print(f"[{idx+1:2d}/50] {label:12s} | {word_count:3d} words | {hypothesis[:60]}...")
    
    # Write JSONL
    output_path = '/Users/steven/code/legal-kit/data/sft_reasoning/results/batch_12.jsonl'
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n✓ Wrote {len(results)} results to {output_path}")
    
    # Statistics
    word_counts = [r['reasoning_word_count'] for r in results]
    print(f"\nWord count statistics:")
    print(f"  Min: {min(word_counts)}")
    print(f"  Max: {max(word_counts)}")
    print(f"  Avg: {sum(word_counts)/len(word_counts):.1f}")
    print(f"  Target range: 50-150 words")
    
    # Check if all in range
    out_of_range = [i for i, wc in enumerate(word_counts) if wc < 50 or wc > 150]
    if out_of_range:
        print(f"\n⚠ {len(out_of_range)} items outside target range: {out_of_range}")
    else:
        print(f"\n✓ All items within target range (50-150 words)")

if __name__ == '__main__':
    main()
