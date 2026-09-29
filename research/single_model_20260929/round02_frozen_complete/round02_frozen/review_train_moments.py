"""Independent algebra/provenance review of saved TRAIN moments, no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b):assert np.isclose(a,b,rtol=1e-10,atol=1e-11),(a,b)
def main():
    p=ROOT/'TRAIN_CALIBRATION_AUDIT.json';r=json.loads(p.read_text())
    assert r['passed'] and r['valid_labels']==1203550 and r['zero_labels']==22646
    assert not r['validation_or_test_inference'] and r['decoded_validation_or_test_label_rows']==0
    assert r['fit_updates']==0 and not r['raw_arrays_exported'] and r['frozen_weights_and_buffers_unchanged']
    assert r['frozen_manifest_sha256']==sha(ROOT/'FROZEN_MANIFEST.json')
    assert r['cache_receipt_sha256']==sha(ROOT/'CACHE_COMPLETE.json')
    variance=json.loads((ROOT/'VARIANCE_PROVENANCE.json').read_text())
    assert r['training_index_sha256']==variance['train_index_sha256']
    for name,value in r['source_hashes'].items():assert sha(name)==value
    for record in [r['pooled']]+r['per_state']:
        n=record['count'];m=record['moments'];sp=m['sum_prediction'];sy=m['sum_label']
        vp=m['sum_prediction_squared']-sp*sp/n
        vy=m['sum_label_squared']-sy*sy/n
        cov=m['sum_cross_product']-sp*sy/n
        sse=m['sum_prediction_squared']+m['sum_label_squared']-2*m['sum_cross_product']
        close(record['native_sse'],sse);close(record['native_r2'],1-sse/vy)
        close(record['native_mse'],sse/n);close(record['unconstrained_affine_slope'],cov/vp)
        close(record['unconstrained_affine_intercept'],sy/n-cov/vp*sp/n)
        assert not record['affine_applied_or_scored']
    for key in r['pooled']['moments']:
        close(r['pooled']['moments'][key],sum(v['moments'][key] for v in r['per_state']))
    close(r['pooled']['label_population_variance'],variance['variance'])
    close(r['pooled']['mean_label'],variance['mean'])
    report={'passed':True,'scope':'Source and saved FP64 moment algebra independently checked; no second prediction pass.',
        'train_native_r2':r['pooled']['native_r2'],'train_OLS_slope':r['pooled']['unconstrained_affine_slope'],
        'train_OLS_intercept':r['pooled']['unconstrained_affine_intercept'],
        'historical_validation_coefficients':r['historical_fixed_validation_coefficients'],
        'all_valid_train_labels_and_zeros_retained':True,'no_coefficients_applied_or_scored':True,
        'no_new_fit_GPU_or_model_inference_by_reviewer':True,
        'interpretation':'TRAIN residual relation differs from previously measured validation relation. This motivates, but does not prove, an in-sample-feature distribution hypothesis; no causal conclusion or deployment improvement follows.',
        'historical_comparability':'Historical affine_clip uses ordinary least squares followed by nonnegative slope and output clipping. Both pooled coefficient pairs here are positive, so clipping is inactive for nonnegative native f.',
        'input_sha256':sha(p),'script_sha256':sha(__file__)}
    (ROOT/'TRAIN_CALIBRATION_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'slope':report['train_OLS_slope'],'intercept':report['train_OLS_intercept']}))
if __name__=='__main__':main()
