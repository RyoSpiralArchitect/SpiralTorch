// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::collections::HashMap;

use st_core::ecosystem::CloudConnector;

/// Groups cloud connectors into provider-specific buckets so metadata tagging
/// stays consistent across pipeline and trainer telemetry.
#[derive(Debug, Default, Clone)]
pub struct CloudTargetSummary {
    azure_targets: Vec<String>,
    aws_targets: Vec<String>,
}

impl CloudTargetSummary {
    /// Builds a summary from the provided connectors, normalising Azure and AWS
    /// targets into stable descriptor strings.
    pub fn from_targets(targets: &[CloudConnector]) -> Self {
        let mut summary = Self::default();
        summary.extend_from_targets(targets);
        summary
    }

    fn extend_from_targets(&mut self, targets: &[CloudConnector]) {
        for target in targets {
            match target {
                CloudConnector::AzureEventHub { namespace, hub } => {
                    self.azure_targets
                        .push(format!("event_hub:{namespace}/{hub}"));
                }
                CloudConnector::AzureStorageQueue { account, queue } => {
                    self.azure_targets
                        .push(format!("storage_queue:{account}/{queue}"));
                }
                CloudConnector::AwsKinesis { region, stream } => {
                    self.aws_targets.push(format!("kinesis:{region}/{stream}"));
                }
                CloudConnector::AwsSqs { region, queue } => {
                    self.aws_targets.push(format!("sqs:{region}/{queue}"));
                }
            }
        }
    }

    /// Extends a metadata vector used by connector events with cloud target
    /// descriptors when any are configured.
    pub fn extend_vec(&self, metadata: &mut Vec<(String, String)>) {
        if let Some(value) = self.azure_value() {
            metadata.push(("azure_targets".to_string(), value));
        }
        if let Some(value) = self.aws_value() {
            metadata.push(("aws_targets".to_string(), value));
        }
    }

    /// Extends a metadata map used by connector events with cloud target
    /// descriptors when any are configured.
    pub fn extend_map(&self, metadata: &mut HashMap<String, String>) {
        if let Some(value) = self.azure_value() {
            metadata.insert("azure_targets".to_string(), value);
        }
        if let Some(value) = self.aws_value() {
            metadata.insert("aws_targets".to_string(), value);
        }
    }

    fn azure_value(&self) -> Option<String> {
        if self.azure_targets.is_empty() {
            None
        } else {
            Some(self.azure_targets.join(","))
        }
    }

    fn aws_value(&self) -> Option<String> {
        if self.aws_targets.is_empty() {
            None
        } else {
            Some(self.aws_targets.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_summary_skips_metadata() {
        let summary = CloudTargetSummary::from_targets(&[]);
        let mut vec_meta = Vec::new();
        let mut map_meta = HashMap::new();

        summary.extend_vec(&mut vec_meta);
        summary.extend_map(&mut map_meta);

        assert!(vec_meta.is_empty());
        assert!(map_meta.is_empty());
    }

    #[test]
    fn summary_formats_targets_consistently() {
        let summary = CloudTargetSummary::from_targets(&[
            CloudConnector::AzureEventHub {
                namespace: "spiral".into(),
                hub: "coord".into(),
            },
            CloudConnector::AzureStorageQueue {
                account: "spiralstorage".into(),
                queue: "jobs".into(),
            },
            CloudConnector::AwsKinesis {
                region: "us-east-2".into(),
                stream: "telemetry".into(),
            },
            CloudConnector::AwsSqs {
                region: "us-west-2".into(),
                queue: "tasks".into(),
            },
        ]);

        let mut vec_meta = Vec::new();
        let mut map_meta = HashMap::new();
        summary.extend_vec(&mut vec_meta);
        summary.extend_map(&mut map_meta);

        assert_eq!(
            vec_meta,
            vec![
                (
                    "azure_targets".to_string(),
                    "event_hub:spiral/coord,storage_queue:spiralstorage/jobs".to_string(),
                ),
                (
                    "aws_targets".to_string(),
                    "kinesis:us-east-2/telemetry,sqs:us-west-2/tasks".to_string(),
                ),
            ]
        );

        assert_eq!(
            map_meta.get("azure_targets"),
            Some(&"event_hub:spiral/coord,storage_queue:spiralstorage/jobs".to_string())
        );
        assert_eq!(
            map_meta.get("aws_targets"),
            Some(&"kinesis:us-east-2/telemetry,sqs:us-west-2/tasks".to_string())
        );
    }
}
