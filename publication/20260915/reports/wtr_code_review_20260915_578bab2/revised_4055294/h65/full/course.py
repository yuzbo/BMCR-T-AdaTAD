"""Provenance checks for the explicitly requested corrected BMCR 20+60 course."""
import math
from pathlib import Path

REVISION = 'lr_identity_crop_validity_v1'
BMCR80_RECIPE = 'bmcr_corrected_warm20_joint60_v1'
COMPONENT_ORDER = ['fixed-assignment classification', 'rematched true-time localization']


def warm_origin(payload, checkpoint, backbone):
    metadata = payload['metadata']
    expected = dict(backbone=backbone, variant='h65', phase='warm', train_videos=200,
                    updates_per_epoch=100, phase_epochs=20, full_training=True,
                    terminal_state='state_dict_ema', fidelity_revision=REVISION)
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('BMCR80 requires the corresponding corrected full warm20 EMA')
    if payload['completed_epochs'] != 20 or payload['successful_updates'] != 2000 or 'state_dict_ema' not in payload:
        raise ValueError('BMCR80 warm checkpoint is not complete20/2000 with EMA')
    return dict(checkpoint=str(Path(checkpoint).resolve()), state_key='state_dict_ema',
                backbone=backbone, completed_epochs=20, successful_updates=2000,
                fidelity_revision=REVISION, train_videos=200,
                recognition_initialization=metadata['initialization'])


def validate_scales(audit, origin):
    if (audit.get('recipe') != BMCR80_RECIPE or audit.get('warm_origin') != origin or
            audit.get('component_order') != COMPONENT_ORDER):
        raise ValueError('BMCR80 utility scales must come from this corrected warm and audit')
    scales = audit['scales']
    if len(scales) != 2 or not all(math.isfinite(v) and v > 0 for v in scales):
        raise ValueError('Utility component scales must be finite and positive')
    return scales


def validate_resume(metadata, total_epochs, phase, variant, backbone, origin=None):
    expected = dict(fidelity_revision=REVISION, backbone=backbone, phase=phase, variant=variant,
                    phase_epochs=20 if phase == 'warm' else total_epochs-20)
    if any(metadata.get(k) != v for k,v in expected.items()):
        raise ValueError('Resume checkpoint belongs to a different training course')
    if total_epochs == 80 and (metadata.get('recipe') != BMCR80_RECIPE or
                              metadata.get('course_total_epochs') != 80 or metadata.get('warm_origin') != origin):
        raise ValueError('BMCR80 cannot resume a different recipe or warm parent')
