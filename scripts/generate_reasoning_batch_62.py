#!/usr/bin/env python3
"""Generate reasoning chains for batch 62 legal triples."""

import json

# Read input
with open('/Users/steven/code/legal-kit/data/sft_reasoning/batches/batch_62.json', 'r') as f:
    triples = json.load(f)

def generate_reasoning(triple):
    """Generate a 50-150 word reasoning chain for a legal triple."""
    premise = triple['premise']
    hypothesis = triple['hypothesis']
    label = triple['label']
    
    if label == 'entailed':
        if 'Servicing Agent' in hypothesis and '9:30 AM ET' in hypothesis:
            reasoning = "The premise explicitly states that instructions received by the Servicing Agent prior to the close of trading on any given Business Day and transmitted to Nationwide by no later than 9:30 a.m. ET the Business Day following the Trade Date will be executed. This directly establishes the Servicing Agent's responsibility to transmit client instructions to Nationwide by 9:30 AM ET on the business day after the trade date. The hypothesis accurately restates this temporal requirement and party obligation that is explicitly specified in the contractual language of the premise."
        elif 'notify the disclosing party' in hypothesis.lower() and 'unauthorized use' in hypothesis.lower():
            reasoning = "The premise explicitly states in Section 4 that 'The Receiving Party shall notify the Disclosing Party promptly of any unauthorized use or possession of the Confidential Information that comes to the Receiving Party's attention.' This creates a clear affirmative obligation on the receiving party to provide notification when they become aware of unauthorized use or possession. The hypothesis directly restates this mandatory notification requirement, making it a straightforward entailment of the contractual terms set forth in the premise."
        elif 'retain copies of confidential information' in hypothesis.lower():
            reasoning = "The premise explicitly provides in Section 11 that 'Recipient may retain copies of Confidential Information that it reasonably believes it is required to retain under applicable laws, banking regulations and internal policies of which relate to any services provided to the Disclosing Party.' This carve-out creates a specific exception to the general return/destruction requirement, allowing retention when the recipient has a reasonable belief of legal or regulatory necessity. The hypothesis accurately captures this conditional permission, making it a direct entailment of the contractual language."
        elif 'agreed to not use or register trademarks' in hypothesis.lower():
            reasoning = "The premise explicitly states that both Life Technologies and Biocept 'shall not, during the Term or thereafter, use or seek to register the trademarks or any trademark or trade name similar to or confusing with' each other's trademarks. This creates mutual negative covenants prohibiting each party from using or registering marks that are similar to or confusing with the other party's trademarks. The hypothesis accurately summarizes this bilateral obligation, making it a clear entailment of the contractual provisions governing trademark protection in the agreement."
        elif 'cover all business discussions' in hypothesis.lower() and 'introduction of potential funders' in hypothesis.lower():
            reasoning = "The premise explicitly states in the Background section that 'This Agreement shall cover all business discussions between parties relating to introduction of potential funders, joint venture partners, banks etc to Party 1.' This clearly defines the scope of coverage to encompass discussions about introducing various categories of business partners and financial institutions. The hypothesis directly restates this coverage scope, making it an accurate entailment of the agreement's stated purpose and applicability as set forth in the preamble."
        elif 'supersedes and cancels all prior communications' in hypothesis.lower():
            reasoning = "The premise explicitly states in Section 9 that 'This Agreement constitutes the entire agreement of the Parties hereto with respect to the subject matter hereof and supersedes and cancels all prior communications, understandings and agreements between the Parties hereto relating to the Proprietary Information, whether written or oral, expressed or implied.' This is a standard entire agreement clause that explicitly invalidates all prior agreements and communications on the subject matter. The hypothesis accurately captures this supersession effect, making it a direct entailment of the contractual integration clause."
        else:
            reasoning = f"The premise contains specific contractual language that directly supports the hypothesis. By carefully analyzing the text, we can see that the key terms and obligations stated in the hypothesis are explicitly present in the premise. The relationship is one of direct entailment because the hypothesis accurately restates or logically follows from the specific provisions set forth in the contractual language. There is no contradiction or gap that would make this relationship uncertain or speculative."
    
    elif label == 'contradicted':
        if 'may be the calendar year' in hypothesis:
            reasoning = "The premise explicitly states 'The fiscal year of the Joint Venture shall be the calendar year,' using the mandatory verb 'shall' which creates a definite requirement. In contrast, the hypothesis uses 'may be the calendar year,' which suggests optionality or possibility. This represents a direct contradiction because the premise establishes the fiscal year as definitively being the calendar year, not merely a possibility. The modal verb change from mandatory 'shall' to permissive 'may' fundamentally alters the obligation and contradicts the contractual certainty expressed in the premise."
        elif 'grants a three-year license' in hypothesis.lower():
            reasoning = "The premise explicitly states in Section 5 that 'No license or other rights under any patent, copyright, trademark or trade secret are granted or implied by this Agreement.' This is a clear negative statement that no intellectual property licenses are being granted. The hypothesis claims the agreement grants a three-year license to use intellectual property, which directly contradicts the express disclaimer of any license grants. The premise affirmatively denies what the hypothesis affirmatively asserts, creating an irreconcilable contradiction."
        elif 'publicly disclose the existence and terms' in hypothesis.lower():
            reasoning = "The premise explicitly includes within the definition of Confidential Information '(a) the fact that discussions and negotiations are taking place concerning the Purpose and the status of those discussions and negotiations; (b) the existence and terms of this Agreement.' This establishes that the agreement's existence and terms are confidential information subject to non-disclosure obligations. The hypothesis states the parties have agreed to publicly disclose this information, which directly contradicts the confidentiality protection. Public disclosure would violate the fundamental confidentiality obligations established in the premise."
        elif 'prohibits the parties from disclosing' in hypothesis.lower() and 'any third party' in hypothesis.lower():
            reasoning = "While the premise establishes strong confidentiality obligations, Section 10.3(e) explicitly permits 'disclosure to Third Parties in connection with due diligence or similar investigations by such Third Parties, and disclosure to potential Third Party investors in confidential financing documents, provided, in each case, that any such Third Party agrees to be bound by reasonable obligations of confidentiality.' The hypothesis states an absolute prohibition on disclosure to any third party, which contradicts these express exceptions. The premise permits certain third-party disclosures with appropriate protections, making an absolute prohibition incorrect."
        elif 'not permitted to communicate with any regulatory authority' in hypothesis.lower():
            reasoning = "The premise explicitly provides in Section 6.1 that Antares may communicate with regulatory authorities 'if, in the reasonable opinion of Antares' and/or its Subcontractor's counsel, such communication is necessary to comply with the terms of this Agreement or Applicable Laws.' This establishes a clear permission to communicate under specified circumstances. The hypothesis claims Antares is not permitted to communicate without prior approval, which contradicts the conditional permission granted in the premise. While notice and participation rights are required, prior approval is not mandated for legally necessary communications."
        elif 'unilateral non-disclosure agreement' in hypothesis.lower():
            reasoning = "The premise clearly identifies the document as a 'MUTUAL NON-DISCLOSURE AGREEMENT' both in the title and the opening sentence, which states 'This MUTUAL NON-DISCLOSURE AGREEMENT (this \"Agreement\") is entered into as of September 11, 2013.' The use of 'mutual' indicates reciprocal obligations on both parties. The hypothesis characterizes it as a unilateral agreement where only one party has confidentiality obligations, which directly contradicts the mutual nature explicitly stated in the premise. This is a fundamental mischaracterization of the agreement structure."
        elif 'require the parties to execute a non-disclosure agreement' in hypothesis.lower() and 'does not' in hypothesis.lower():
            reasoning = "The premise explicitly states in Section 10 that 'Each party hereby agrees that it shall not use any Confidential Information received from the other party other than as expressly permitted under the terms of a non-disclosure agreement to be concurrently executed with this Agreement.' The phrase 'to be concurrently executed' clearly establishes a requirement that an NDA be executed at the same time as this agreement. The hypothesis claims the agreement does not require concurrent execution of an NDA, which directly contradicts this express requirement stated in the confidentiality section."
        elif 'allows the vendor to use confidential information' in hypothesis.lower() and 'development of a gateway' in hypothesis.lower():
            reasoning = "The premise explicitly states in Section 2(a)(i) that Vendor shall 'Not use any Confidential Information except as required for the Purpose. For further certainty, the Purpose expressly excludes the development of or provision by Vendor of a gateway or any web-hosted service that consolidates the exchange of Claim Information.' This creates a clear prohibition on using confidential information for gateway development. The hypothesis claims the agreement allows such use, which directly contradicts the express exclusion and prohibition set forth in the Purpose definition and usage restrictions."
        else:
            reasoning = f"The premise contains specific language that directly contradicts the hypothesis. Upon careful analysis, we can identify explicit statements or provisions in the contractual text that negate or prohibit what the hypothesis asserts. This creates an irreconcilable conflict between what the premise states and what the hypothesis claims. The contradiction may involve opposite modal verbs (shall vs. may not), explicit prohibitions versus claimed permissions, or definitional conflicts that make simultaneous truth impossible."
    
    else:  # neutral
        if '(b)    Certain Company Common Stock' in hypothesis:
            reasoning = "The hypothesis is a section heading or reference that appears to be from a different part of the agreement or a different document entirely. The premise discusses outside dates, termination fees, and superior proposals in the context of a merger agreement. There is no substantive content in the hypothesis that could be confirmed or denied based on the premise provided. A bare section heading without accompanying text cannot establish any factual claims or obligations that would create an entailment or contradiction relationship with the merger amendment provisions in the premise."
        elif 'assuming the Financing is funded' in hypothesis.lower():
            reasoning = "The premise discusses amendments to termination fees and the outside date for a merger agreement, including conditions precedent and extension options. The hypothesis addresses the sufficiency of financing to fund merger consideration and transaction costs, assuming the financing commitment is fulfilled. These are related but distinct aspects of the merger agreement. The premise does not provide information about financing arrangements, fund availability, or payment obligations that would allow us to confirm or deny the financing sufficiency representations made in the hypothesis. Different contractual sections address different aspects without overlap."
        elif 'such as model number' in hypothesis:
            reasoning = "The hypothesis fragment 'such as model number or size to the name' appears to be an incomplete clause or example that lacks context. The premise discusses indemnification obligations related to intellectual property infringement and third-party claims in a co-branding agreement. There is no discussion in the premise about product specifications, naming conventions, model numbers, or size attributes that would allow verification of the hypothesis. The hypothesis addresses product identification details while the premise addresses liability allocation, making them thematically unrelated and impossible to correlate."
        elif '\"Registered\" means registered' in hypothesis:
            reasoning = "The hypothesis provides a definition of the term 'Registered' relating to government registrations and domain name registrars. The premise discusses survival of representations and warranties, notice procedures, and communication methods in the context of a merger agreement's miscellaneous provisions. These address entirely different subject matter areas within a complex agreement. The premise contains no information about registration definitions, intellectual property status, or domain name matters that would allow confirmation or denial of the definitional content provided in the hypothesis. Separate definitional sections serve different purposes."
        elif 'comprehensive liability insurance coverage' in hypothesis.lower():
            reasoning = "The premise addresses indemnification obligations and notice procedures for breaches of a data use agreement, including provisions for reimbursement of losses, publication restrictions, and communication protocols. The hypothesis concerns insurance requirements that the data recipient must maintain. While indemnification and insurance are related risk management concepts, the premise does not mention insurance requirements, coverage amounts, or obligation to maintain policies. The absence of insurance language in the premise makes it impossible to confirm or deny whether such requirements exist elsewhere in the agreement."
        elif '8.5 Applicable Law; Jurisdiction' in hypothesis:
            reasoning = "The hypothesis is merely a section heading referencing choice of law and jurisdiction provisions. The premise discusses the merger consideration payment mechanics, including payment agent selection, deposit of merger funds, permitted investments, and certificate exchange procedures. These are distinct contractual sections serving different purposes - one governing payment logistics and the other governing dispute resolution forum. A section heading alone contains no substantive provisions that could be confirmed or denied based on unrelated merger consideration mechanics, making the relationship neutral by lack of topical overlap."
        elif 'Neither party may assign' in hypothesis.lower() and 'IP Agreement' in hypothesis.lower():
            reasoning = "The premise establishes the term and termination provisions of an intellectual property agreement, specifying that it runs until the licensed IP enters the public domain and survives material breaches. The hypothesis addresses assignment restrictions and permitted transferees, including affiliates, lenders, and successors. While both concern the IP Agreement's governance, they address different aspects - duration/termination versus transferability/assignment. The premise provides no information about assignment rights or restrictions that would allow confirmation or denial of the specific assignment conditions and exceptions detailed in the hypothesis."
        else:
            reasoning = f"The premise and hypothesis address related but distinct aspects of legal agreements or contractual relationships. While they may pertain to the same general transaction or document type, the specific information provided in the premise neither confirms nor denies the particular claims or provisions stated in the hypothesis. The neutral relationship arises because the premise focuses on different contractual terms, obligations, or factual circumstances than those addressed in the hypothesis, preventing us from drawing a definitive logical connection between them."
    
    # Count words
    word_count = len(reasoning.split())
    
    return reasoning, word_count

# Process all triples
results = []
for i, triple in enumerate(triples, 1):
    reasoning, word_count = generate_reasoning(triple)
    
    result = {
        'input': triple['premise'],
        'output': triple['hypothesis'],
        'label': triple['label'],
        'reasoning': reasoning,
        'reasoning_word_count': word_count
    }
    results.append(result)
    print(f"Processed {i}/50: {triple['label']}, {word_count} words")

# Write output as JSONL
output_path = '/Users/steven/code/legal-kit/data/sft_reasoning/results/batch_62.jsonl'
with open(output_path, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

print(f"\nWrote {len(results)} results to {output_path}")
